using System;
using AniML.World;
using UnityEngine;

namespace AniML.Creatures
{
    /// <summary>
    /// Closed-loop RL locomotion motor for the 38-DOF PolyArt Deer.
    ///
    /// Ticks the trained Isaac Lab AMP+PPO actor network (D2 / D2b,
    /// [122 -&gt; 512 -&gt; 256 -&gt; 128 -&gt; 34] ELU MLP) at 50 Hz (dt = 0.02s),
    /// integrates the coupled 85-D multibody + hoof-contact state, drives
    /// the 31-bone Malbers PolyArt Deer rig via <see cref="DeerRigDriver"/>,
    /// and conforms root motion to <see cref="ProceduralWorld"/> terrain.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(DeerRigDriver))]
    public sealed class DeerPolicyLocomotionMotor : MonoBehaviour, ICreatureMotor
    {
        public const float PolicyDt = 0.02f; // 50 Hz control frequency (sim.dt=0.005 * decimation=4)

        [Header("Policy")]
        [Tooltip("Resource path under Assets/_Project/Resources/ (without extension).")]
        public string PolicyResourcePath = "Policies/deer_polyart_d2";

        [Header("Terrain & Navigation")]
        [Tooltip("Maximum water depth (m) the deer will wade through (fords are ~0.25-0.45m).")]
        public float MaxWadeDepth = 0.55f;

        [Tooltip("Maximum terrain slope (degrees) the deer will climb.")]
        public float MaxClimbSlopeDeg = 36f;

        [Tooltip("Rate at which the root aligns to terrain slope normals.")]
        public float TerrainAlignSpeed = 8f;

        private CreatureController _controller;
        private DeerPolicyBundle _bundle;
        private DeerRigDriver _rig;

        // Preallocated buffers for zero-GC 50 Hz evaluation
        private readonly float[] _obs = new float[DeerPolicyBundle.ObsDim];
        private readonly float[] _actions = new float[DeerPolicyBundle.ActDim];
        private readonly float[] _lastActions = new float[DeerPolicyBundle.ActDim];
        private readonly float[] _state = new float[DeerPolicyBundle.StateDim];
        private readonly float[] _h0 = new float[512];
        private readonly float[] _h1 = new float[256];
        private readonly float[] _h2 = new float[128];
        private readonly float[] _physInput = new float[120];
        private readonly float[] _physPred = new float[DeerPolicyBundle.StateDim];
        private readonly float[] _basis = new float[DeerPolicyBundle.BasisDim];
        private readonly float[] _manifold = new float[DeerPolicyBundle.StateDim];

        // Interpolation buffers between 50 Hz physics ticks
        private readonly float[] _prevJointPos = new float[DeerPolicyBundle.JointCount];
        private readonly float[] _currJointPos = new float[DeerPolicyBundle.JointCount];
        private readonly float[] _renderJointPos = new float[DeerPolicyBundle.JointCount];

        private float _stepAccumulator;
        private float _stridePhaseRad;
        private float _idleClock;
        private float _locomotionBlend01;
        private float _smoothedTurnBendRad;
        private float _smoothedBrowsePitchRad;
        private float _currentSpeedMps;
        private Vector3 _smoothedGroundNormal = Vector3.up;
        private bool _initialized;

        /// <summary>
        /// Set by <see cref="DeerEcologyBrain"/> when the deer stops to graze or drink.
        /// Lowers the neck and head toward the ground/water during idle.
        /// </summary>
        public bool IsBrowsingOrDrinking { get; set; }

        /// <summary>Current forward locomotion speed in world space (m/s) — satisfies <see cref="ICreatureMotor"/>.</summary>
        public float CurrentSpeed => _currentSpeedMps;

        /// <summary>Current forward locomotion speed in world space (m/s).</summary>
        public float CurrentSpeedMps => _currentSpeedMps;

        /// <summary>Read-only access to the latest 122-D observation vector (for diagnostics).</summary>
        public ReadOnlySpan<float> LatestObservation => _obs;

        /// <summary>Read-only access to the latest 34-D policy action vector (for diagnostics).</summary>
        public ReadOnlySpan<float> LatestActions => _actions;

        private void Awake()
        {
            EnsureInitialized();
        }

        public void Attach(CreatureController controller)
        {
            _controller = controller;
            EnsureInitialized();
        }

        public void EnsureInitialized()
        {
            if (_initialized)
                return;

            _bundle = DeerPolicyBundle.Load(PolicyResourcePath);
            if (_bundle == null)
                return;

            _rig = GetComponent<DeerRigDriver>();
            if (_rig == null)
                _rig = gameObject.AddComponent<DeerRigDriver>();
            _rig.Initialize(_bundle, GetHashCode());

            // Initialize state on the slow-walk manifold with a per-instance phase offset
            // so herds don't step in robotic lockstep.
            _stridePhaseRad = Mathf.Abs(GetHashCode() * 0.371f) % (2f * Mathf.PI);
            _idleClock = _stridePhaseRad;
            _locomotionBlend01 = 0.85f;
            _bundle.EvaluateManifold(1.5f, _stridePhaseRad, _basis, _state);
            Array.Copy(_state, 9, _currJointPos, 0, DeerPolicyBundle.JointCount);
            Array.Copy(_currJointPos, 0, _prevJointPos, 0, DeerPolicyBundle.JointCount);

            // Apply initial natural stride pose immediately so even before Play mode starts
            // (or on frame 0) the rig is posed cleanly.
            var projGravInit = new Vector3(_state[6], _state[7], _state[8]);
            _rig.ApplyPose(_currJointPos, projGravInit, 0f, 0f, 0f);

            _initialized = true;
        }

        public void Apply(in CreatureIntent intent, float deltaTime)
        {
            var body = _controller != null ? _controller.transform : transform;
            var def = _controller != null ? _controller.Definition : null;
            var world = _controller != null ? _controller.World : FindAnyObjectByType<ProceduralWorld>();
            Tick(body, def, world, in intent, deltaTime);
        }

        public void Tick(Transform body, CreatureDefinition def, ProceduralWorld world,
            in CreatureIntent intent, float dt)
        {
            EnsureInitialized();
            if (!_initialized || _bundle == null || dt <= 0f)
                return;

            dt = Mathf.Min(dt, 0.1f);
            _idleClock += dt;

            float turnRate = def != null ? def.TurnRate : 220f;
            float maxSpeed = def != null ? def.MaxSpeed : 3.2f;

            // 1. Determine desired heading & turn rate in world XZ
            var flatDir = new Vector3(intent.MoveDirection.x, 0f, intent.MoveDirection.z);
            float speed01 = Mathf.Clamp01(intent.Speed01);
            bool wantsToMove = speed01 > 0.04f && flatDir.sqrMagnitude > 1e-4f;

            float yawRateRad = 0f;
            if (flatDir.sqrMagnitude > 1e-4f)
            {
                Vector3 targetFwd = flatDir.normalized;
                Vector3 currentFwd = Vector3.ProjectOnPlane(body.forward, Vector3.up).normalized;
                if (currentFwd.sqrMagnitude < 1e-4f)
                    currentFwd = Vector3.forward;

                float signedAngleDeg = Vector3.SignedAngle(currentFwd, targetFwd, Vector3.up);
                float maxTurnDeg = turnRate * dt;
                float appliedTurnDeg = Mathf.Clamp(signedAngleDeg, -maxTurnDeg, maxTurnDeg);
                yawRateRad = (appliedTurnDeg * Mathf.Deg2Rad) / Mathf.Max(dt, 1e-4f);

                float newYaw = body.eulerAngles.y + appliedTurnDeg;
                body.rotation = Quaternion.Euler(0f, newYaw, 0f);
            }

            // Smooth spine/neck lateral bend into turns (positive yawRate in Unity +Y is right turn -> -Z_I yaw in Isaac)
            float targetTurnBend = Mathf.Clamp(-yawRateRad * 0.22f, -0.45f, 0.45f);
            _smoothedTurnBendRad = Mathf.Lerp(_smoothedTurnBendRad, targetTurnBend, 8f * dt);

            // Smooth neck/head pitch when grazing or drinking while stopped
            float targetBrowsePitch = (!wantsToMove && IsBrowsingOrDrinking)
                ? 0.52f + 0.08f * Mathf.Sin(_idleClock * 3.2f)
                : 0f;
            _smoothedBrowsePitchRad = Mathf.Lerp(_smoothedBrowsePitchRad, targetBrowsePitch, 5f * dt);

            // 2. Step the 50 Hz Closed-Loop Neural Policy & Multibody Integrator
            float cmdVx = wantsToMove ? Mathf.Lerp(0.95f, 3.0f, speed01) : 0f;
            float cmdVy = 0f;
            float cmdWz = Mathf.Clamp(-yawRateRad, -1.0f, 1.0f);

            _locomotionBlend01 = Mathf.MoveTowards(_locomotionBlend01, wantsToMove ? 1f : 0f, 3.5f * dt);

            _stepAccumulator += dt;
            int maxSubSteps = 4;
            while (_stepAccumulator >= PolicyDt && maxSubSteps-- > 0)
            {
                _stepAccumulator -= PolicyDt;
                Array.Copy(_currJointPos, 0, _prevJointPos, 0, DeerPolicyBundle.JointCount);

                if (_locomotionBlend01 > 0.01f)
                {
                    StepPolicyClosedLoop(Mathf.Max(0.95f, cmdVx), cmdVy, cmdWz);
                }

                // Blend active stride pose with calm breathing/browsing idle pose when stopping
                for (int j = 0; j < DeerPolicyBundle.JointCount; j++)
                {
                    float activeQ = _state[9 + j];
                    float idleQ = SampleIdleJointPose(j, _idleClock);
                    _currJointPos[j] = Mathf.Lerp(idleQ, activeQ, _locomotionBlend01);
                }
            }

            // 3. Interpolate 38-DOF joint positions for sub-frame rendering smoothness
            float alpha = Mathf.Clamp01(_stepAccumulator / PolicyDt);
            for (int j = 0; j < DeerPolicyBundle.JointCount; j++)
                _renderJointPos[j] = Mathf.Lerp(_prevJointPos[j], _currJointPos[j], alpha);

            // 4. Advance world-space root position along terrain with passability safety checks
            float rawPolicyVx = _state[0] * _rig.ModelScale;
            float desiredWorldSpeed = wantsToMove
                ? Mathf.Lerp(rawPolicyVx, maxSpeed * speed01, 0.35f) * _locomotionBlend01
                : 0f;
            _currentSpeedMps = Mathf.Lerp(_currentSpeedMps, desiredWorldSpeed, 10f * dt);

            Vector3 pos = body.position;
            if (_currentSpeedMps > 0.01f)
            {
                Vector3 fwdFlat = Vector3.ProjectOnPlane(body.forward, Vector3.up).normalized;
                Vector3 candidatePos = pos + fwdFlat * (_currentSpeedMps * dt);

                if (IsPositionPassable(world, candidatePos))
                {
                    pos = candidatePos;
                }
                else
                {
                    // Try deflecting 40 degrees left or right around obstacle/water/cliff
                    Vector3 leftDir = Quaternion.Euler(0f, -42f, 0f) * fwdFlat;
                    Vector3 rightDir = Quaternion.Euler(0f, 42f, 0f) * fwdFlat;
                    Vector3 leftCand = pos + leftDir * (_currentSpeedMps * dt);
                    Vector3 rightCand = pos + rightDir * (_currentSpeedMps * dt);

                    if (IsPositionPassable(world, leftCand))
                    {
                        pos = leftCand;
                        body.rotation = Quaternion.LookRotation(leftDir, Vector3.up);
                    }
                    else if (IsPositionPassable(world, rightCand))
                    {
                        pos = rightCand;
                        body.rotation = Quaternion.LookRotation(rightDir, Vector3.up);
                    }
                    else
                    {
                        _currentSpeedMps = 0f;
                    }
                }
            }

            // 5. Conform root height and pitch/roll to ProceduralWorld terrain
            if (world != null)
            {
                pos.y = world.SampleHeight(pos.x, pos.z);
                Vector3 groundNormal = world.SampleNormal(pos.x, pos.z);
                _smoothedGroundNormal = Vector3.Slerp(_smoothedGroundNormal, groundNormal, TerrainAlignSpeed * dt).normalized;

                Vector3 fwdOnSlope = Vector3.ProjectOnPlane(body.forward, _smoothedGroundNormal).normalized;
                if (fwdOnSlope.sqrMagnitude > 1e-4f)
                    body.rotation = Quaternion.LookRotation(fwdOnSlope, _smoothedGroundNormal);
            }
            else
            {
                pos.y = 0f;
            }

            body.position = pos;

            // 6. Apply 38-DOF pose + pelvis stride bob to the 31-bone skeleton
            var projGravIsaac = new Vector3(
                _state[6] * _locomotionBlend01,
                _state[7] * _locomotionBlend01,
                Mathf.Lerp(-1f, _state[8], _locomotionBlend01));

            float pelvisBob = Mathf.Clamp(_state[2] * 0.018f * _locomotionBlend01, -0.045f, 0.045f);
            _rig.ApplyPose(_renderJointPos, projGravIsaac, pelvisBob, _smoothedTurnBendRad, _smoothedBrowsePitchRad);
        }

        private void StepPolicyClosedLoop(float cmdVx, float cmdVy, float cmdWz)
        {
            // Assemble 122-D observation vector matching PolicyCfg in training/polyart/amp_env_cfg.py:
            //   [0..2]    base_lin_vel (3)
            //   [3..5]    base_ang_vel (3)
            //   [6..8]    projected_gravity (3)
            //   [9..11]   velocity_commands (3)
            //   [12..49]  joint_pos (38, USD BFS order)
            //   [50..87]  joint_vel (38, USD BFS order)
            //   [88..121] last_action (34, active joints in USD BFS order)
            _obs[0] = _state[0];
            _obs[1] = _state[1];
            _obs[2] = _state[2];

            _obs[3] = _state[3];
            _obs[4] = _state[4];
            _obs[5] = _state[5];

            _obs[6] = _state[6];
            _obs[7] = _state[7];
            _obs[8] = _state[8];

            _obs[9] = cmdVx;
            _obs[10] = cmdVy;
            _obs[11] = cmdWz;

            Array.Copy(_state, 9, _obs, 12, DeerPolicyBundle.JointCount);
            Array.Copy(_state, 47, _obs, 50, DeerPolicyBundle.JointCount);
            Array.Copy(_lastActions, 0, _obs, 88, DeerPolicyBundle.ActDim);

            // Evaluate 4-layer ELU actor network: _obs[122] -> _actions[34]
            _bundle.EvaluatePolicy(_obs, _actions, _h0, _h1, _h2);

            // Advance stride phase at speed-matched gait frequency
            float freqHz = _bundle.SampleStrideFrequencyHz(cmdVx);
            _stridePhaseRad = (_stridePhaseRad + 2f * Mathf.PI * freqHz * PolicyDt) % (2f * Mathf.PI);

            // Step coupled multibody + contact-rebound physics state
            _bundle.StepClosedLoopPhysics(
                _state, _actions, cmdVx, _stridePhaseRad,
                _physInput, _physPred, _basis, _manifold);

            Array.Copy(_actions, 0, _lastActions, 0, DeerPolicyBundle.ActDim);
        }

        private float SampleIdleJointPose(int jointIdx, float t)
        {
            // Subtle breathing, ear flicks, and tail swish while standing/grazing
            return jointIdx switch
            {
                3  => 0.18f + 0.10f * Mathf.Sin(t * 2.1f), // Pelvis_to_Tail
                9  => 0.03f * Mathf.Sin(t * 1.6f),         // Spine_to_Spine1 (breathing)
                36 => 0.04f * Mathf.Sin(t * 3.7f),         // Head_to_L_Ear
                37 => 0.04f * Mathf.Cos(t * 3.3f),         // Head_to_R_Ear
                _  => 0f,
            };
        }

        private bool IsPositionPassable(ProceduralWorld world, Vector3 candidatePos)
        {
            if (world == null || world.Settings == null)
                return true;

            float half = world.Settings.WorldSize * 0.47f;
            if (Mathf.Abs(candidatePos.x) > half || Mathf.Abs(candidatePos.z) > half)
                return false;

            float h = world.SampleHeight(candidatePos.x, candidatePos.z);
            float waterSurf = world.SampleWaterSurface(candidatePos.x, candidatePos.z);
            float waterDepth = waterSurf - h;
            if (waterDepth > MaxWadeDepth)
                return false;

            // Avoid ocean below sea level unless shallow beach/ford
            if (h < world.Settings.SeaLevel - 0.25f)
                return false;

            Vector3 normal = world.SampleNormal(candidatePos.x, candidatePos.z);
            float slopeDeg = Mathf.Acos(Mathf.Clamp(normal.y, -1f, 1f)) * Mathf.Rad2Deg;
            return slopeDeg <= MaxClimbSlopeDeg;
        }
    }
}
