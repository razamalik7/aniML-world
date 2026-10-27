using AniML.World;
using UnityEngine;

namespace AniML.Creatures
{
    /// <summary>
    /// Ecosystem-aware brain for the PolyArt Deer on <see cref="ProceduralWorld"/>.
    ///
    /// Samples the island's procedural habitat layers (<c>SampleBrowse01</c>,
    /// <c>SampleWaterAccess01</c>, <c>SampleCover01</c>, river fords, slope, and
    /// water depth) to autonomously forage in meadows, drink at freshwater banks,
    /// cross rivers at gravel-bar fords, and rest under forest canopy.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class DeerEcologyBrain : MonoBehaviour, ICreatureBrain
    {
        public enum EcologyState
        {
            Roaming,
            Foraging,
            Grazing,
            SeekingWater,
            Drinking,
            RestingInCover,
        }

        [Header("Behavior Tuning")]
        [Tooltip("Normalized speed [0..1] during relaxed foraging/roaming (maps to ~1.4-1.8 m/s).")]
        [Range(0.1f, 1f)]
        public float CruiseSpeed01 = 0.42f;

        [Tooltip("Normalized speed [0..1] when trotting purposefully toward water or cover (~2.4 m/s).")]
        [Range(0.2f, 1f)]
        public float TrotSpeed01 = 0.72f;

        [Tooltip("Seconds between waypoint re-evaluations while moving.")]
        public float DecisionInterval = 2.8f;

        [Header("Live Ecological State (Read-Only)")]
        [SerializeField] private EcologyState _state = EcologyState.Foraging;
        [SerializeField, Range(0f, 1f)] private float _hunger01 = 0.45f;
        [SerializeField, Range(0f, 1f)] private float _thirst01 = 0.25f;
        [SerializeField, Range(0f, 1f)] private float _fatigue01 = 0.15f;

        private CreatureController _controller;
        private Vector3 _targetPosition;
        private Vector3 _currentHeading = Vector3.forward;
        private float _decisionTimer;
        private float _dwellTimer;
        private DeerPolicyLocomotionMotor _policyMotor;
        private bool _initialized;

        public EcologyState State => _state;
        public float Hunger01 => _hunger01;
        public float Thirst01 => _thirst01;
        public float Fatigue01 => _fatigue01;

        private void Awake()
        {
            _policyMotor = GetComponent<DeerPolicyLocomotionMotor>();
            // Stagger initial needs slightly per individual so a herd exhibits varied behavior
            uint hash = (uint)Mathf.Abs(GetHashCode() * 2654435761u);
            _hunger01 = 0.30f + (hash & 0xFF) / 255f * 0.40f;
            _thirst01 = 0.15f + ((hash >> 8) & 0xFF) / 255f * 0.35f;
            _fatigue01 = 0.10f + ((hash >> 16) & 0xFF) / 255f * 0.25f;
        }

        public void Attach(CreatureController controller)
        {
            _controller = controller;
            if (_policyMotor == null && controller != null)
                _policyMotor = controller.GetComponent<DeerPolicyLocomotionMotor>();
        }

        public CreatureIntent Think(float dt)
        {
            var body = _controller != null ? _controller.transform : transform;
            var pos = body.position;
            var world = _controller != null ? _controller.World : FindAnyObjectByType<ProceduralWorld>();

            if (!_initialized)
            {
                _currentHeading = Vector3.ProjectOnPlane(body.forward, Vector3.up).normalized;
                if (_currentHeading.sqrMagnitude < 1e-4f)
                    _currentHeading = Vector3.forward;
                _targetPosition = pos + _currentHeading * 18f;
                _initialized = true;
            }

            // 1. Update internal physiological needs
            _hunger01 = Mathf.Clamp01(_hunger01 + 0.012f * dt);
            _thirst01 = Mathf.Clamp01(_thirst01 + 0.010f * dt);
            _fatigue01 = Mathf.Clamp01(_fatigue01 + 0.006f * dt);

            float browseHere = world != null ? world.SampleBrowse01(pos.x, pos.z) : 0.6f;
            float waterHere = world != null ? world.SampleWaterAccess01(pos.x, pos.z) : 0.3f;
            float coverHere = world != null ? world.SampleCover01(pos.x, pos.z) : 0.4f;

            // 2. Handle stationary dwell states (Grazing / Drinking / Resting)
            if (_state == EcologyState.Grazing || _state == EcologyState.Drinking || _state == EcologyState.RestingInCover)
            {
                _dwellTimer -= dt;
                if (_state == EcologyState.Grazing)
                    _hunger01 = Mathf.Clamp01(_hunger01 - 0.09f * dt);
                else if (_state == EcologyState.Drinking)
                    _thirst01 = Mathf.Clamp01(_thirst01 - 0.14f * dt);
                else if (_state == EcologyState.RestingInCover)
                    _fatigue01 = Mathf.Clamp01(_fatigue01 - 0.08f * dt);

                if (_policyMotor != null)
                    _policyMotor.IsBrowsingOrDrinking = (_state == EcologyState.Grazing || _state == EcologyState.Drinking);

                if (_dwellTimer > 0f)
                {
                    return new CreatureIntent
                    {
                        MoveDirection = _currentHeading,
                        Speed01 = 0f,
                    };
                }

                // Finished dwelling -> transition back to active exploration
                if (_policyMotor != null)
                    _policyMotor.IsBrowsingOrDrinking = false;
                SelectNextActiveState();
                _decisionTimer = 0f;
            }

            // 3. Check if we arrived at a rich ecological resource patch and should stop to graze/drink/rest
            Vector3 toTarget = _targetPosition - pos;
            toTarget.y = 0f;
            float distToTarget = toTarget.magnitude;

            if (_state == EcologyState.Foraging && browseHere > 0.45f && (_hunger01 > 0.35f || distToTarget < 4f))
            {
                if (Random.value < 0.35f * dt || distToTarget < 3.5f)
                {
                    _state = EcologyState.Grazing;
                    _dwellTimer = Random.Range(3.5f, 6.5f);
                    if (_policyMotor != null) _policyMotor.IsBrowsingOrDrinking = true;
                    return new CreatureIntent { MoveDirection = _currentHeading, Speed01 = 0f };
                }
            }
            else if (_state == EcologyState.SeekingWater && waterHere > 0.45f)
            {
                _state = EcologyState.Drinking;
                _dwellTimer = Random.Range(3.0f, 5.5f);
                if (_policyMotor != null) _policyMotor.IsBrowsingOrDrinking = true;
                return new CreatureIntent { MoveDirection = _currentHeading, Speed01 = 0f };
            }
            else if (_state == EcologyState.Roaming && coverHere > 0.65f && _fatigue01 > 0.65f)
            {
                _state = EcologyState.RestingInCover;
                _dwellTimer = Random.Range(4.0f, 7.0f);
                return new CreatureIntent { MoveDirection = _currentHeading, Speed01 = 0f };
            }

            // 4. Periodically evaluate habitat around the deer and pick the best waypoint
            _decisionTimer -= dt;
            if (_decisionTimer <= 0f || distToTarget < 3.0f)
            {
                SelectNextActiveState();
                _targetPosition = PickBestHabitatWaypoint(world, pos, _currentHeading);
                _decisionTimer = DecisionInterval * Random.Range(0.85f, 1.15f);
            }

            Vector3 desiredDir = _targetPosition - pos;
            desiredDir.y = 0f;
            if (desiredDir.sqrMagnitude > 1e-3f)
                _currentHeading = desiredDir.normalized;

            float speed = (_state == EcologyState.SeekingWater && _thirst01 > 0.55f)
                ? TrotSpeed01
                : CruiseSpeed01;

            return new CreatureIntent
            {
                MoveDirection = _currentHeading,
                Speed01 = speed,
            };
        }

        private void SelectNextActiveState()
        {
            if (_thirst01 > 0.60f && _thirst01 >= _hunger01)
                _state = EcologyState.SeekingWater;
            else if (_hunger01 > 0.35f)
                _state = EcologyState.Foraging;
            else
                _state = EcologyState.Roaming;
        }

        private Vector3 PickBestHabitatWaypoint(ProceduralWorld world, Vector3 origin, Vector3 currentFwd)
        {
            if (world == null || world.Settings == null)
            {
                float angle = Random.Range(-55f, 55f);
                Vector3 dir = Quaternion.Euler(0f, angle, 0f) * currentFwd;
                return origin + dir.normalized * 20f;
            }

            const int Directions = 12;
            float[] radii = { 16f, 32f };
            Vector3 bestTarget = origin + currentFwd * 14f;
            float bestScore = float.NegativeInfinity;

            for (int i = 0; i < Directions; i++)
            {
                float angleDeg = (i / (float)Directions) * 360f + Random.Range(-10f, 10f);
                Vector3 dir = Quaternion.Euler(0f, angleDeg, 0f) * Vector3.forward;

                foreach (float r in radii)
                {
                    Vector3 candidate = origin + dir * r;
                    if (!IsSegmentPassable(world, origin, candidate))
                        continue;

                    float browse = world.SampleBrowse01(candidate.x, candidate.z);
                    float water = world.SampleWaterAccess01(candidate.x, candidate.z);
                    float cover = world.SampleCover01(candidate.x, candidate.z);

                    // Alignment with current heading (encourages smooth, natural trails)
                    float forwardAlignment = Vector3.Dot(dir, currentFwd); // [-1..1]

                    float score = forwardAlignment * 0.35f + Random.Range(-0.08f, 0.08f);
                    switch (_state)
                    {
                        case EcologyState.SeekingWater:
                            score += water * 2.2f + browse * 0.4f;
                            break;
                        case EcologyState.Foraging:
                            score += browse * 1.8f + cover * 0.35f + water * 0.25f;
                            break;
                        default:
                            score += browse * 0.9f + cover * 0.8f + water * 0.4f;
                            break;
                    }

                    // Bonus for gravel-bar river fords when near a river channel
                    foreach (var ford in world.Fords)
                    {
                        float dFord = Vector2.Distance(new Vector2(candidate.x, candidate.z), ford.Pos);
                        if (dFord < ford.Radius * 2.5f)
                            score += 0.35f;
                    }

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestTarget = candidate;
                    }
                }
            }

            return bestTarget;
        }

        private static bool IsSegmentPassable(ProceduralWorld world, Vector3 start, Vector3 end)
        {
            float half = world.Settings.WorldSize * 0.46f;
            if (Mathf.Abs(end.x) > half || Mathf.Abs(end.z) > half)
                return false;

            const int Samples = 4;
            for (int s = 1; s <= Samples; s++)
            {
                float t = s / (float)Samples;
                float x = Mathf.Lerp(start.x, end.x, t);
                float z = Mathf.Lerp(start.z, end.z, t);

                float h = world.SampleHeight(x, z);
                if (h < world.Settings.SeaLevel - 0.15f)
                    return false;

                float waterDepth = world.SampleWaterSurface(x, z) - h;
                if (waterDepth > 0.48f)
                    return false;

                Vector3 n = world.SampleNormal(x, z);
                float slopeDeg = Mathf.Acos(Mathf.Clamp(n.y, -1f, 1f)) * Mathf.Rad2Deg;
                if (slopeDeg > 34f)
                    return false;
            }

            return true;
        }
    }
}
