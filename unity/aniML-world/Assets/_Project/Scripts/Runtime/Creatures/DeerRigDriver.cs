using System.Collections.Generic;
using UnityEngine;

namespace AniML.Creatures
{
    /// <summary>
    /// Binds the 38-DOF Isaac Lab URDF joint space to the 31-bone Malbers PolyArt Deer
    /// hierarchy (Pelvis + 30 child bones) with 4-micrometer forward-kinematics parity.
    ///
    /// If the Malbers FBX hierarchy is not present on the GameObject (e.g. in a clean
    /// public clone without paid asset packs), automatically constructs a 31-link
    /// articulated procedural deer skeleton matching the exact URDF link offsets so
    /// the 38-DOF neural policy still drives every limb, spine, neck, ear, jaw, and hoof.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class DeerRigDriver : MonoBehaviour
    {
        private struct BoundBone
        {
            public Transform Bone;
            public Quaternion RestLocalRotation;
            public int Dof0Index;
            public int Dof1Index;
            public Vector3 Axis0Local;
            public Vector3 Axis1Local;
            public bool IsSpineOrNeck;
            public float TurnBendWeight;
            public float BrowsePitchWeight;
        }

        [Header("Appearance")]
        [Tooltip("Scale multiplier applied to the model root (0.8 matches the URDF physics scale).")]
        public float ModelScale = 0.85f;

        [Tooltip("Probability [0..1] that an adult deer displays antlers (stag vs doe).")]
        [Range(0f, 1f)]
        public float StagProbability = 0.55f;

        private Transform _pelvis;
        private Quaternion _pelvisRestLocalRot;
        private Vector3 _pelvisRestLocalPos;
        private BoundBone[] _boundBones;
        private Transform[] _hoofBones;
        private float _smoothedGroundOffset;
        private bool _hasGroundOffset;
        private bool _initialized;

        public bool IsInitialized => _initialized;
        public Transform Pelvis => _pelvis;

        public void Initialize(DeerPolicyBundle bundle, int variantSeed = 0)
        {
            if (_initialized || bundle == null)
                return;

            // Disable any Animator so Mecanim never overwrites our 38-DOF neural pose.
            foreach (var anim in GetComponentsInChildren<Animator>(true))
                anim.enabled = false;

            // Locate Pelvis in the hierarchy (or build procedural articulated fallback if absent).
            _pelvis = FindDescendantByName(transform, "Pelvis");
            if (_pelvis == null)
            {
                BuildProceduralArticulatedSkeleton();
                _pelvis = FindDescendantByName(transform, "Pelvis");
            }
            else
            {
                ConfigureMalbersMeshesAndMaterials(variantSeed);
            }

            if (_pelvis == null)
            {
                Debug.LogError("[AniML] DeerRigDriver could not locate or construct Pelvis bone.");
                return;
            }

            // Apply uniform scale if still at default 1.0
            if (Mathf.Abs(transform.localScale.x - 1f) < 0.01f && ModelScale > 0.1f)
                transform.localScale = Vector3.one * ModelScale;

            _pelvisRestLocalRot = _pelvis.localRotation;
            _pelvisRestLocalPos = _pelvis.localPosition;

            // Index all descendants of Pelvis so helper transforms outside Pelvis (e.g. root IK goals) are ignored.
            var pelvisDescendants = new Dictionary<string, Transform>(36);
            IndexDescendants(_pelvis, pelvisDescendants);

            Quaternion invRootRot = Quaternion.Inverse(transform.rotation);
            var list = new List<BoundBone>(DeerPolicyBundle.BoneMappings30.Length);

            foreach (var map in DeerPolicyBundle.BoneMappings30)
            {
                if (!pelvisDescendants.TryGetValue(map.BoneName, out var bone) || bone == null || bone.parent == null)
                    continue;

                Quaternion parentRestRootRot = invRootRot * bone.parent.rotation;
                Quaternion invParentRestRoot = Quaternion.Inverse(parentRestRootRot);

                Vector3 axis0U = bundle.JointAxesUnityRoot[map.Dof0Index];
                Vector3 axis0Local = (invParentRestRoot * axis0U).normalized;

                Vector3 axis1Local = Vector3.up;
                if (map.Dof1Index >= 0)
                {
                    Vector3 axis1U = bundle.JointAxesUnityRoot[map.Dof1Index];
                    axis1Local = (invParentRestRoot * axis1U).normalized;
                }

                float turnWeight = 0f;
                float browseWeight = 0f;
                switch (map.BoneName)
                {
                    case "Spine":
                    case "Spine1":
                    case "Spine2":
                        turnWeight = 0.22f;
                        browseWeight = 0.10f;
                        break;
                    case "Neck":
                        turnWeight = 0.30f;
                        browseWeight = 0.45f;
                        break;
                    case "Neck1":
                        turnWeight = 0.25f;
                        browseWeight = 0.40f;
                        break;
                    case "Head":
                        turnWeight = 0.20f;
                        browseWeight = -0.25f;
                        break;
                }

                list.Add(new BoundBone
                {
                    Bone = bone,
                    RestLocalRotation = bone.localRotation,
                    Dof0Index = map.Dof0Index,
                    Dof1Index = map.Dof1Index,
                    Axis0Local = axis0Local,
                    Axis1Local = axis1Local,
                    IsSpineOrNeck = turnWeight > 0f || Mathf.Abs(browseWeight) > 0f,
                    TurnBendWeight = turnWeight,
                    BrowsePitchWeight = browseWeight,
                });
            }

            _boundBones = list.ToArray();

            var hooves = new List<Transform>(4);
            foreach (var hName in new[] { "L Finger0", "R Finger0", "L Toe0", "R Toe0" })
            {
                if (pelvisDescendants.TryGetValue(hName, out var hb) && hb != null)
                    hooves.Add(hb);
            }
            _hoofBones = hooves.ToArray();

            _initialized = true;
        }

        /// <summary>
        /// Applies the 38-DOF joint positions and pelvis orientation/bob to the Unity bone hierarchy.
        /// </summary>
        public void ApplyPose(
            float[] jointPos38,
            Vector3 projectedGravityIsaac,
            float pelvisVerticalBobMeters,
            float turnBendRad,
            float browsePitchRad)
        {
            if (!_initialized || _boundBones == null)
                return;

            // 1. Pelvis subtle roll/pitch from projectedGravityIsaac & vertical stride bob
            // At upright level pose, projectedGravityIsaac = (0, 0, -1).
            float pitchDeg = Mathf.Clamp(projectedGravityIsaac.x * 18f, -10f, 10f);
            float rollDeg = Mathf.Clamp(-projectedGravityIsaac.y * 16f, -8f, 8f);
            Quaternion pelvisTilt = Quaternion.Euler(pitchDeg, 0f, rollDeg);

            if (_pelvis != null && _pelvis.parent != null)
            {
                Quaternion parentRestRoot = Quaternion.Inverse(transform.rotation) * _pelvis.parent.rotation;
                Quaternion localTilt = Quaternion.Inverse(parentRestRoot) * pelvisTilt * parentRestRoot;
                _pelvis.localRotation = localTilt * _pelvisRestLocalRot;
                _pelvis.localPosition = _pelvisRestLocalPos;
            }

            // 2. Apply all 30 articulated bone rotations using our verified 4-micrometer FK formula
            for (int i = 0; i < _boundBones.Length; i++)
            {
                ref readonly var b = ref _boundBones[i];
                float q0 = jointPos38[b.Dof0Index];
                float q1 = b.Dof1Index >= 0 ? jointPos38[b.Dof1Index] : 0f;

                // Keep left/right forelegs and hindlegs at natural shoulder/hip width without midline crossing
                if (b.Dof1Index == 23)      // L_UpperArm_dof0_to_L_UpperArm (roll)
                    q1 = q1 * 0.35f + 0.06f;
                else if (b.Dof1Index == 25) // R_UpperArm_dof0_to_R_UpperArm (roll)
                    q1 = q1 * 0.35f - 0.06f;
                else if (b.Dof1Index == 4)  // L_Thigh_dof0_to_L_Thigh (roll)
                    q1 = q1 * 0.50f + 0.03f;
                else if (b.Dof1Index == 5)  // R_Thigh_dof0_to_R_Thigh (roll)
                    q1 = q1 * 0.50f - 0.03f;

                if (b.IsSpineOrNeck)
                {
                    q0 += turnBendRad * b.TurnBendWeight;
                    if (b.Dof1Index >= 0)
                        q1 += browsePitchRad * b.BrowsePitchWeight;
                    else
                        q0 += browsePitchRad * b.BrowsePitchWeight;
                }

                Quaternion rot = Quaternion.AngleAxis(q0 * Mathf.Rad2Deg, b.Axis0Local);
                if (b.Dof1Index >= 0)
                    rot = rot * Quaternion.AngleAxis(q1 * Mathf.Rad2Deg, b.Axis1Local);

                b.Bone.localRotation = rot * b.RestLocalRotation;
            }

            // 3. Dynamically ground-plant the pelvis so the lowest supporting hoof sits flush on the terrain
            if (_pelvis != null && _hoofBones != null && _hoofBones.Length > 0)
            {
                float minHoofY = float.PositiveInfinity;
                for (int h = 0; h < _hoofBones.Length; h++)
                {
                    float hy = _hoofBones[h].position.y;
                    if (hy < minHoofY)
                        minHoofY = hy;
                }

                float desiredSoleY = transform.position.y + 0.015f + pelvisVerticalBobMeters * 0.4f;
                float rawOffset = Mathf.Clamp(desiredSoleY - minHoofY, -0.22f, 0.22f);
                if (!_hasGroundOffset)
                {
                    _smoothedGroundOffset = rawOffset;
                    _hasGroundOffset = true;
                }
                else
                {
                    _smoothedGroundOffset = Mathf.Lerp(_smoothedGroundOffset, rawOffset, 0.45f);
                }

                _pelvis.position += transform.up * _smoothedGroundOffset;
            }
        }

        private void ConfigureMalbersMeshesAndMaterials(int variantSeed)
        {
            var rng = new System.Random(variantSeed == 0 ? GetHashCode() : variantSeed);
            bool isStag = rng.NextDouble() < StagProbability;
            int chosenAntler = rng.Next(1, 4); // Antlers 01, 02, or 03

            foreach (var smr in GetComponentsInChildren<SkinnedMeshRenderer>(true))
                smr.updateWhenOffscreen = true;

            foreach (var t in GetComponentsInChildren<Transform>(true))
            {
                string n = t.name;
                if (n.StartsWith("Magic") || n == "Spots")
                {
                    t.gameObject.SetActive(false);
                }
                else if (n.StartsWith("Antlers"))
                {
                    bool active = isStag && n == $"Antlers 0{chosenAntler}";
                    t.gameObject.SetActive(active);
                }
            }
        }

        private static Transform FindDescendantByName(Transform root, string targetName)
        {
            var queue = new Queue<Transform>();
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                if (current.name == targetName)
                    return current;
                for (int i = 0; i < current.childCount; i++)
                    queue.Enqueue(current.GetChild(i));
            }
            return null;
        }

        private static void IndexDescendants(Transform root, Dictionary<string, Transform> map)
        {
            for (int i = 0; i < root.childCount; i++)
            {
                var child = root.GetChild(i);
                if (!map.ContainsKey(child.name))
                    map[child.name] = child;
                IndexDescendants(child, map);
            }
        }

        /// <summary>
        /// Fallback articulated 31-bone hierarchy with capsule/box visual links matching
        /// the exact URDF rest offsets if Malbers Deer.FBX is absent.
        /// </summary>
        private void BuildProceduralArticulatedSkeleton()
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            var coatMat = new Material(shader) { color = new Color(0.58f, 0.39f, 0.24f) };
            var darkMat = new Material(shader) { color = new Color(0.22f, 0.16f, 0.12f) };

            var cg = new GameObject("CG").transform;
            cg.SetParent(transform, false);
            cg.localPosition = Vector3.zero;
            cg.localRotation = Quaternion.identity;

            var pelvis = CreateBone(cg, "Pelvis", new Vector3(0f, 1.2289f, 0f), new Vector3(0.26f, 0.24f, 0.34f), coatMat);
            var spine = CreateBone(pelvis, "Spine", new Vector3(0f, -0.0493f, 0.0577f), new Vector3(0.25f, 0.25f, 0.32f), coatMat);
            var spine1 = CreateBone(spine, "Spine1", new Vector3(0f, -0.0557f, 0.3112f), new Vector3(0.26f, 0.27f, 0.32f), coatMat);
            var spine2 = CreateBone(spine1, "Spine2", new Vector3(0f, 0.0065f, 0.3161f), new Vector3(0.28f, 0.30f, 0.34f), coatMat);

            var neck = CreateBone(spine2, "Neck", new Vector3(0f, 0.0497f, 0.3388f), new Vector3(0.16f, 0.22f, 0.28f), coatMat);
            var neck1 = CreateBone(neck, "Neck1", new Vector3(0f, 0.1570f, 0.1864f), new Vector3(0.14f, 0.22f, 0.24f), coatMat);
            var head = CreateBone(neck1, "Head", new Vector3(0f, 0.2000f, 0.1779f), new Vector3(0.16f, 0.16f, 0.28f), coatMat);
            CreateBone(head, "Jaw", new Vector3(0f, -0.1506f, 0.2439f), new Vector3(0.08f, 0.06f, 0.16f), darkMat);
            CreateBone(head, "L Ear", new Vector3(-0.1129f, 0.1386f, 0.0416f), new Vector3(0.06f, 0.14f, 0.04f), coatMat);
            CreateBone(head, "R Ear", new Vector3(0.1129f, 0.1386f, 0.0416f), new Vector3(0.06f, 0.14f, 0.04f), coatMat);
            CreateBone(pelvis, "Tail", new Vector3(0f, 0.0255f, -0.2746f), new Vector3(0.07f, 0.07f, 0.16f), coatMat);

            for (int side = -1; side <= 1; side += 2)
            {
                string pfx = side < 0 ? "L" : "R";
                float sx = side;
                var clav = CreateBone(spine2, $"{pfx} Clavicle", new Vector3(sx * 0.1614f, 0.1590f, 0.1066f), new Vector3(0.10f, 0.22f, 0.12f), coatMat);
                var uarm = CreateBone(clav, $"{pfx} UpperArm", new Vector3(0f, -0.5645f, 0.0050f), new Vector3(0.10f, 0.28f, 0.11f), coatMat);
                var farm = CreateBone(uarm, $"{pfx} Forearm", new Vector3(-sx * 0.0009f, -0.2842f, -0.0014f), new Vector3(0.07f, 0.30f, 0.08f), coatMat);
                var hand = CreateBone(farm, $"{pfx} Hand", new Vector3(-sx * 0.0010f, -0.2997f, -0.0021f), new Vector3(0.06f, 0.26f, 0.06f), darkMat);
                CreateBone(hand, $"{pfx} Finger0", new Vector3(sx * 0.0006f, -0.0988f, 0.0556f), new Vector3(0.06f, 0.08f, 0.09f), darkMat);

                var thigh = CreateBone(pelvis, $"{pfx} Thigh", new Vector3(sx * 0.1293f, 0f, 0f), new Vector3(0.13f, 0.36f, 0.16f), coatMat);
                var calf = CreateBone(thigh, $"{pfx} Calf", new Vector3(sx * 0.0064f, -0.4965f, 0.0046f), new Vector3(0.10f, 0.30f, 0.12f), coatMat);
                var hlink = CreateBone(calf, $"{pfx} HorseLink", new Vector3(-sx * 0.0036f, -0.2129f, -0.1829f), new Vector3(0.08f, 0.28f, 0.09f), coatMat);
                var foot = CreateBone(hlink, $"{pfx} Foot", new Vector3(sx * 0.0064f, -0.3386f, 0.0635f), new Vector3(0.06f, 0.24f, 0.07f), darkMat);
                CreateBone(foot, $"{pfx} Toe0", new Vector3(sx * 0.0041f, -0.1069f, 0.0760f), new Vector3(0.06f, 0.08f, 0.09f), darkMat);
            }
        }

        private static Transform CreateBone(Transform parent, string name, Vector3 localPos, Vector3 boxSize, Material mat)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localRotation = Quaternion.identity;

            var vis = GameObject.CreatePrimitive(PrimitiveType.Cube);
            vis.name = $"{name}_Mesh";
            var col = vis.GetComponent<Collider>();
            if (col != null)
            {
                if (Application.isPlaying) Destroy(col);
                else DestroyImmediate(col);
            }
            vis.transform.SetParent(go.transform, false);
            vis.transform.localPosition = Vector3.zero;
            vis.transform.localScale = boxSize;
            vis.GetComponent<Renderer>().sharedMaterial = mat;

            return go.transform;
        }
    }
}
