using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace AniML.Creatures
{
    /// <summary>
    /// Runtime neural policy &amp; PhysX-calibrated multibody dynamics bundle for
    /// the 38-DOF PolyArt Deer (D2 / D2b AMP+PPO checkpoints).
    ///
    /// Loads the binary bundle exported by tools/export_unity_policy.py from
    /// Resources/Policies/deer_polyart_d2.bytes (or deer_polyart_d2b.bytes).
    /// Runs the 4-layer ELU actor MLP ([122 -&gt; 512 -&gt; 256 -&gt; 128 -&gt; 34]) and
    /// coupled multibody contact-stabilized state transition in pure C# with
    /// zero per-tick GC allocations.
    /// </summary>
    public sealed class DeerPolicyBundle
    {
        public const int ExpectedMagic = 0x4C4D4E41; // 'ANML'
        public const int ExpectedVersion = 1;
        public const int ObsDim = 122;
        public const int ActDim = 34;
        public const int JointCount = 38;
        public const int StateDim = 85;
        public const int Harmonics = 6;
        public const int BasisDim = 1 + 2 * Harmonics; // 13

        /// <summary>
        /// Exact PhysX / Isaac Lab internal joint ordering (USD BFS traversal from Pelvis).
        /// Matches robot.data.joint_names during D2 / D2b training.
        /// </summary>
        public static readonly string[] JointNames38 =
        {
            "Pelvis_to_L_Thigh_dof0",        //  0
            "Pelvis_to_R_Thigh_dof0",        //  1
            "Pelvis_to_Spine_dof0",          //  2
            "Pelvis_to_Tail",                //  3 (passive)
            "L_Thigh_dof0_to_L_Thigh",       //  4
            "R_Thigh_dof0_to_R_Thigh",       //  5
            "Spine_dof0_to_Spine",           //  6
            "L_Thigh_to_L_Calf",             //  7
            "R_Thigh_to_R_Calf",             //  8
            "Spine_to_Spine1",               //  9
            "L_Calf_to_L_HorseLink",         // 10
            "R_Calf_to_R_HorseLink",         // 11
            "Spine1_to_Spine2",              // 12
            "L_HorseLink_to_L_Foot",         // 13
            "R_HorseLink_to_R_Foot",         // 14
            "Spine2_to_L_Clavicle",          // 15
            "Spine2_to_Neck_dof0",           // 16
            "Spine2_to_R_Clavicle",          // 17
            "L_Foot_to_L_Toe0",              // 18
            "R_Foot_to_R_Toe0",              // 19
            "L_Clavicle_to_L_UpperArm_dof0", // 20
            "Neck_dof0_to_Neck",             // 21
            "R_Clavicle_to_R_UpperArm_dof0", // 22
            "L_UpperArm_dof0_to_L_UpperArm", // 23
            "Neck_to_Neck1_dof0",            // 24
            "R_UpperArm_dof0_to_R_UpperArm", // 25
            "L_UpperArm_to_L_Forearm",       // 26
            "Neck1_dof0_to_Neck1",           // 27
            "R_UpperArm_to_R_Forearm",       // 28
            "L_Forearm_to_L_Hand",           // 29
            "Neck1_to_Head_dof0",            // 30
            "R_Forearm_to_R_Hand",           // 31
            "L_Hand_to_L_Finger0",           // 32
            "Head_dof0_to_Head",             // 33
            "R_Hand_to_R_Finger0",           // 34
            "Head_to_Jaw",                   // 35 (passive)
            "Head_to_L_Ear",                 // 36 (passive)
            "Head_to_R_Ear",                 // 37 (passive)
        };

        public readonly struct BoneJointMap
        {
            public readonly string BoneName;
            public readonly int Dof0Index;
            public readonly int Dof1Index;

            public BoneJointMap(string boneName, int dof0, int dof1 = -1)
            {
                BoneName = boneName;
                Dof0Index = dof0;
                Dof1Index = dof1;
            }
        }

        /// <summary>
        /// Maps each of the 30 articulated child bones under Pelvis in Malbers Deer.FBX
        /// to its 1 or 2 DOF indices in JointNames38.
        /// </summary>
        public static readonly BoneJointMap[] BoneMappings30 =
        {
            new BoneJointMap("Spine",       2,  6),
            new BoneJointMap("Spine1",      9, -1),
            new BoneJointMap("Spine2",     12, -1),
            new BoneJointMap("Neck",       16, 21),
            new BoneJointMap("Neck1",      24, 27),
            new BoneJointMap("Head",       30, 33),
            new BoneJointMap("Jaw",        35, -1),
            new BoneJointMap("L Ear",      36, -1),
            new BoneJointMap("R Ear",      37, -1),
            new BoneJointMap("L Clavicle", 15, -1),
            new BoneJointMap("L UpperArm", 20, 23),
            new BoneJointMap("L Forearm",  26, -1),
            new BoneJointMap("L Hand",     29, -1),
            new BoneJointMap("L Finger0",  32, -1),
            new BoneJointMap("R Clavicle", 17, -1),
            new BoneJointMap("R UpperArm", 22, 25),
            new BoneJointMap("R Forearm",  28, -1),
            new BoneJointMap("R Hand",     31, -1),
            new BoneJointMap("R Finger0",  34, -1),
            new BoneJointMap("L Thigh",     0,  4),
            new BoneJointMap("L Calf",      7, -1),
            new BoneJointMap("L HorseLink",10, -1),
            new BoneJointMap("L Foot",     13, -1),
            new BoneJointMap("L Toe0",     18, -1),
            new BoneJointMap("R Thigh",     1,  5),
            new BoneJointMap("R Calf",      8, -1),
            new BoneJointMap("R HorseLink",11, -1),
            new BoneJointMap("R Foot",     14, -1),
            new BoneJointMap("R Toe0",     19, -1),
            new BoneJointMap("Tail",        3, -1),
        };

        public sealed class DenseLayer
        {
            public readonly int InDim;
            public readonly int OutDim;
            public readonly float[] Weights; // [OutDim * InDim] row-major
            public readonly float[] Biases;  // [OutDim]

            public DenseLayer(int inDim, int outDim, float[] weights, float[] biases)
            {
                InDim = inDim;
                OutDim = outDim;
                Weights = weights;
                Biases = biases;
            }

            public void Forward(float[] input, float[] output, bool applyElu)
            {
                int inDim = InDim;
                int outDim = OutDim;
                float[] w = Weights;
                float[] b = Biases;

                for (int o = 0; o < outDim; o++)
                {
                    float sum = b[o];
                    int rowOffset = o * inDim;
                    for (int i = 0; i < inDim; i++)
                        sum += w[rowOffset + i] * input[i];

                    if (applyElu && sum < 0f)
                        sum = Mathf.Exp(sum) - 1f;

                    output[o] = sum;
                }
            }
        }

        public readonly int[] ActionJointIds;      // [34]
        public readonly float[] JointLowerLimits;  // [38]
        public readonly float[] JointUpperLimits;  // [38]
        public readonly Vector3[] JointAxesIsaac;  // [38]
        public readonly Vector3[] JointAxesUnityRoot; // [38]: (a_y, -a_z, -a_x)
        public readonly DenseLayer[] Layers;       // [4]
        public readonly int PhysInDim;             // 120
        public readonly int PhysOutDim;            // 85
        public readonly float[] PhysicsW;          // [120 * 85] row-major
        public readonly float[] ManifoldSpeeds;    // [3]
        public readonly float[] ManifoldFreqs;     // [3]
        public readonly float[][] ManifoldCoeffs;  // [3][13 * 85]

        private static readonly Dictionary<string, DeerPolicyBundle> Cache = new();

        public static DeerPolicyBundle Load(string resourcePath = "Policies/deer_polyart_d2")
        {
            if (Cache.TryGetValue(resourcePath, out var cached) && cached != null)
                return cached;

            var asset = Resources.Load<TextAsset>(resourcePath);
            if (asset == null)
            {
                Debug.LogError($"[AniML] Policy bundle not found at Resources/{resourcePath}");
                return null;
            }

            var bundle = Parse(asset.bytes);
            Cache[resourcePath] = bundle;
            return bundle;
        }

        public static DeerPolicyBundle Parse(byte[] bytes)
        {
            using var ms = new MemoryStream(bytes, writable: false);
            using var br = new BinaryReader(ms);

            int magic = br.ReadInt32();
            int version = br.ReadInt32();
            int obsDim = br.ReadInt32();
            int actDim = br.ReadInt32();
            int jointCount = br.ReadInt32();
            int stateDim = br.ReadInt32();
            int harmonics = br.ReadInt32();

            if (magic != ExpectedMagic || version != ExpectedVersion ||
                obsDim != ObsDim || actDim != ActDim || jointCount != JointCount ||
                stateDim != StateDim || harmonics != Harmonics)
            {
                throw new InvalidDataException(
                    $"[AniML] Invalid policy bundle header: magic=0x{magic:X8}, v={version}, obs={obsDim}, act={actDim}, joints={jointCount}");
            }

            int[] actionIds = ReadIntArray(br, actDim);
            float[] lo = ReadFloatArray(br, jointCount);
            float[] hi = ReadFloatArray(br, jointCount);
            float[] axesRaw = ReadFloatArray(br, jointCount * 3);

            var axesIsaac = new Vector3[jointCount];
            var axesUnity = new Vector3[jointCount];
            for (int j = 0; j < jointCount; j++)
            {
                float ax = axesRaw[j * 3 + 0];
                float ay = axesRaw[j * 3 + 1];
                float az = axesRaw[j * 3 + 2];
                axesIsaac[j] = new Vector3(ax, ay, az);
                // Conjugation by improper basis change P: (-y_I, z_I, x_I) with det(P) = -1
                // maps right-handed axis (ax, ay, az) -> left-handed Unity axis (ay, -az, -ax).
                axesUnity[j] = new Vector3(ay, -az, -ax).normalized;
            }

            int numLayers = br.ReadInt32();
            var layers = new DenseLayer[numLayers];
            for (int l = 0; l < numLayers; l++)
            {
                int inDim = br.ReadInt32();
                int outDim = br.ReadInt32();
                float[] w = ReadFloatArray(br, outDim * inDim);
                float[] b = ReadFloatArray(br, outDim);
                layers[l] = new DenseLayer(inDim, outDim, w, b);
            }

            int physIn = br.ReadInt32();
            int physOut = br.ReadInt32();
            float[] physW = ReadFloatArray(br, physIn * physOut);

            float[] speeds = ReadFloatArray(br, 3);
            float[] freqs = ReadFloatArray(br, 3);
            var manifolds = new float[3][];
            for (int m = 0; m < 3; m++)
                manifolds[m] = ReadFloatArray(br, BasisDim * stateDim);

            return new DeerPolicyBundle(
                actionIds, lo, hi, axesIsaac, axesUnity,
                layers, physIn, physOut, physW, speeds, freqs, manifolds);
        }

        private DeerPolicyBundle(
            int[] actionJointIds,
            float[] lo,
            float[] hi,
            Vector3[] axesIsaac,
            Vector3[] axesUnity,
            DenseLayer[] layers,
            int physIn,
            int physOut,
            float[] physW,
            float[] speeds,
            float[] freqs,
            float[][] manifolds)
        {
            ActionJointIds = actionJointIds;
            JointLowerLimits = lo;
            JointUpperLimits = hi;
            JointAxesIsaac = axesIsaac;
            JointAxesUnityRoot = axesUnity;
            Layers = layers;
            PhysInDim = physIn;
            PhysOutDim = physOut;
            PhysicsW = physW;
            ManifoldSpeeds = speeds;
            ManifoldFreqs = freqs;
            ManifoldCoeffs = manifolds;
        }

        /// <summary>
        /// Evaluates the 4-layer ELU actor MLP: obs[122] -&gt; actions[34].
        /// </summary>
        public void EvaluatePolicy(
            float[] obs122,
            float[] actions34,
            float[] h0_512,
            float[] h1_256,
            float[] h2_128)
        {
            Layers[0].Forward(obs122, h0_512, applyElu: true);
            Layers[1].Forward(h0_512, h1_256, applyElu: true);
            Layers[2].Forward(h1_256, h2_128, applyElu: true);
            Layers[3].Forward(h2_128, actions34, applyElu: false);
        }

        /// <summary>
        /// Evaluates the speed-interpolated Fourier contact-rebound manifold at stride phase (rad).
        /// </summary>
        public void EvaluateManifold(float speedMps, float phaseRad, float[] basis13, float[] stateOut85)
        {
            basis13[0] = 1f;
            for (int k = 1; k <= Harmonics; k++)
            {
                float angle = k * phaseRad;
                basis13[2 * k - 1] = Mathf.Cos(angle);
                basis13[2 * k] = Mathf.Sin(angle);
            }

            // Determine interpolation weights across the 3 speed manifolds (slow=1.2, med=1.8, fast=2.6)
            int idxA = 0, idxB = 1;
            float t = 0f;
            if (speedMps <= ManifoldSpeeds[0])
            {
                idxA = idxB = 0;
                t = 0f;
            }
            else if (speedMps <= ManifoldSpeeds[1])
            {
                idxA = 0;
                idxB = 1;
                t = (speedMps - ManifoldSpeeds[0]) / Mathf.Max(0.01f, ManifoldSpeeds[1] - ManifoldSpeeds[0]);
            }
            else if (speedMps <= ManifoldSpeeds[2])
            {
                idxA = 1;
                idxB = 2;
                t = (speedMps - ManifoldSpeeds[1]) / Mathf.Max(0.01f, ManifoldSpeeds[2] - ManifoldSpeeds[1]);
            }
            else
            {
                idxA = idxB = 2;
                t = 0f;
            }

            float[] mA = ManifoldCoeffs[idxA];
            float[] mB = ManifoldCoeffs[idxB];
            float oneMinusT = 1f - t;

            for (int s = 0; s < StateDim; s++)
            {
                float sum = 0f;
                for (int b = 0; b < BasisDim; b++)
                {
                    int offset = b * StateDim + s;
                    float coeff = oneMinusT * mA[offset] + t * mB[offset];
                    sum += basis13[b] * coeff;
                }
                stateOut85[s] = sum;
            }
        }

        /// <summary>
        /// Returns the interpolated stride frequency (Hz) for a given forward speed (m/s).
        /// </summary>
        public float SampleStrideFrequencyHz(float speedMps)
        {
            if (speedMps <= ManifoldSpeeds[0])
                return ManifoldFreqs[0] * Mathf.Clamp(speedMps / ManifoldSpeeds[0], 0.65f, 1f);
            if (speedMps <= ManifoldSpeeds[1])
            {
                float t = (speedMps - ManifoldSpeeds[0]) / (ManifoldSpeeds[1] - ManifoldSpeeds[0]);
                return Mathf.Lerp(ManifoldFreqs[0], ManifoldFreqs[1], t);
            }
            if (speedMps <= ManifoldSpeeds[2])
            {
                float t = (speedMps - ManifoldSpeeds[1]) / (ManifoldSpeeds[2] - ManifoldSpeeds[1]);
                return Mathf.Lerp(ManifoldFreqs[1], ManifoldFreqs[2], t);
            }
            return ManifoldFreqs[2];
        }

        /// <summary>
        /// Steps the 85-D physical state s_t = [rlin(3), rang(3), grav(3), q(38), qd(38)]
        /// over dt = 0.02s using the coupled multibody matrix W(s_t, a_t) blended with the
        /// PhysX ground-contact rebound manifold for long-horizon stability.
        /// </summary>
        public void StepClosedLoopPhysics(
            float[] state85,
            float[] actions34,
            float cmdSpeedMps,
            float phaseRad,
            float[] physInput120,
            float[] physPred85,
            float[] basis13,
            float[] manifold85)
        {
            Array.Copy(state85, 0, physInput120, 0, StateDim);
            for (int a = 0; a < ActDim; a++)
                physInput120[StateDim + a] = Mathf.Clamp(actions34[a], -3.5f, 3.5f);
            physInput120[119] = 1f; // bias term

            // physPred85 = physInput120 @ PhysicsW (where PhysicsW is [120, 85] row-major)
            Array.Clear(physPred85, 0, StateDim);
            float[] w = PhysicsW;
            for (int i = 0; i < PhysInDim; i++)
            {
                float u = physInput120[i];
                if (Mathf.Abs(u) < 1e-8f) continue;
                int rowOffset = i * StateDim;
                for (int o = 0; o < StateDim; o++)
                    physPred85[o] += u * w[rowOffset + o];
            }

            EvaluateManifold(cmdSpeedMps, phaseRad, basis13, manifold85);

            // Blend 65% live coupled physics W(s_t, act_t) + 35% contact rebound manifold
            const float physWeight = 0.65f;
            const float manWeight = 0.35f;
            for (int i = 0; i < StateDim; i++)
                state85[i] = physWeight * physPred85[i] + manWeight * manifold85[i];

            // Clamp root linear & angular velocity
            state85[0] = Mathf.Clamp(0.85f * state85[0] + 0.15f * cmdSpeedMps, 0.2f, 3.8f);
            state85[1] = Mathf.Clamp(state85[1], -1.0f, 1.0f);
            state85[2] = Mathf.Clamp(state85[2], -2.0f, 2.0f);
            state85[3] = Mathf.Clamp(state85[3], -3.0f, 3.0f);
            state85[4] = Mathf.Clamp(state85[4], -3.0f, 3.0f);
            state85[5] = Mathf.Clamp(state85[5], -3.0f, 3.0f);

            // Normalize projected gravity in upright cone
            float gx = state85[6];
            float gy = state85[7];
            float gz = Mathf.Clamp(state85[8], -1.0f, -0.88f);
            float gNorm = Mathf.Sqrt(gx * gx + gy * gy + gz * gz);
            if (gNorm > 1e-5f)
            {
                state85[6] = gx / gNorm;
                state85[7] = gy / gNorm;
                state85[8] = gz / gNorm;
            }
            else
            {
                state85[6] = 0f;
                state85[7] = 0f;
                state85[8] = -1f;
            }

            // Clamp joint positions to URDF limits and joint velocities to [-22, 22] rad/s
            for (int j = 0; j < JointCount; j++)
            {
                state85[9 + j] = Mathf.Clamp(state85[9 + j], JointLowerLimits[j], JointUpperLimits[j]);
                state85[47 + j] = Mathf.Clamp(state85[47 + j], -22f, 22f);
            }
        }

        private static int[] ReadIntArray(BinaryReader br, int count)
        {
            byte[] raw = br.ReadBytes(count * sizeof(int));
            int[] arr = new int[count];
            Buffer.BlockCopy(raw, 0, arr, 0, raw.Length);
            return arr;
        }

        private static float[] ReadFloatArray(BinaryReader br, int count)
        {
            byte[] raw = br.ReadBytes(count * sizeof(float));
            float[] arr = new float[count];
            Buffer.BlockCopy(raw, 0, arr, 0, raw.Length);
            return arr;
        }
    }
}
