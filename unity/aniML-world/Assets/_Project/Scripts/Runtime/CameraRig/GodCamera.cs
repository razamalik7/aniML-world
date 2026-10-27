using AniML.Creatures;
using AniML.World;
using UnityEngine;
using UnityEngine.InputSystem;

namespace AniML.CameraRig
{
    /// <summary>
    /// RTS-style sandbox camera: WASD pan, scroll zoom, hold-right-mouse (or Q/E) orbit,
    /// middle-mouse drag pan, and Tab to cycle-follow individual creatures on the island.
    /// Operates on a terrain-conforming pivot point; the camera looks at the pivot
    /// from (yaw, pitch, distance).
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public sealed class GodCamera : MonoBehaviour
    {
        [Header("Pan")]
        [SerializeField] private float panSpeed = 25f;
        [SerializeField] private float dragPanSpeed = 1.5f;

        [Header("Orbit")]
        [SerializeField] private float orbitSpeed = 0.25f;
        [SerializeField] private float keyOrbitSpeed = 90f;
        [SerializeField] private Vector2 pitchLimits = new Vector2(8f, 82f);

        [Header("Zoom")]
        [SerializeField] private float zoomSpeed = 3f;
        [SerializeField] private Vector2 distanceLimits = new Vector2(2.2f, 180f);

        [Header("State")]
        [SerializeField] private Vector3 pivot = new Vector3(0f, 25.2f, 0f);
        [SerializeField] private float yaw = 45f;
        [SerializeField] private float pitch = 22f;
        [SerializeField] private float distance = 22f;

        private ProceduralWorld _world;
        private CreatureController[] _creatures;
        private int _followIndex = -1;
        private bool _showHud = true;

        private void Start()
        {
            _world = FindAnyObjectByType<ProceduralWorld>();
            if (_world != null)
            {
                _world.EnsureSamplers();
                pivot.y = _world.SampleHeight(pivot.x, pivot.z) + 1.1f;
            }
        }

        private void LateUpdate()
        {
            if (_world == null)
                _world = FindAnyObjectByType<ProceduralWorld>();

            var kb = Keyboard.current;
            var mouse = Mouse.current;
            float dt = Time.unscaledDeltaTime;

            if (kb != null)
            {
                if (kb.hKey.wasPressedThisFrame)
                    _showHud = !_showHud;

                if (kb.tabKey.wasPressedThisFrame)
                {
                    _creatures = FindObjectsByType<CreatureController>(FindObjectsSortMode.None);
                    if (_creatures != null && _creatures.Length > 0)
                    {
                        _followIndex = (_followIndex + 1) % _creatures.Length;
                        distance = Mathf.Clamp(distance, 3.2f, 6.5f);
                        pitch = Mathf.Clamp(pitch, 14f, 26f);
                    }
                }
                if (kb.escapeKey.wasPressedThisFrame)
                {
                    _followIndex = -1;
                }
            }

            if (kb != null && mouse != null)
            {
                // --- Keyboard pan (relative to camera yaw, on the ground plane) ---
                var panInput = Vector2.zero;
                if (kb.wKey.isPressed) panInput.y += 1f;
                if (kb.sKey.isPressed) panInput.y -= 1f;
                if (kb.dKey.isPressed) panInput.x += 1f;
                if (kb.aKey.isPressed) panInput.x -= 1f;

                if (panInput.sqrMagnitude > 0.01f)
                    _followIndex = -1; // manual pan detaches from creature follow

                var yawRot = Quaternion.Euler(0f, yaw, 0f);
                float panScale = panSpeed * Mathf.Max(distance / 40f, 0.25f);
                pivot += yawRot * new Vector3(panInput.x, 0f, panInput.y) * (panScale * dt);

                // --- Middle-mouse drag pan ---
                if (mouse.middleButton.isPressed)
                {
                    _followIndex = -1;
                    var delta = mouse.delta.ReadValue();
                    pivot -= yawRot * new Vector3(delta.x, 0f, delta.y) * (dragPanSpeed * distance * 0.001f);
                }

                // --- Orbit ---
                if (mouse.rightButton.isPressed)
                {
                    var delta = mouse.delta.ReadValue();
                    yaw += delta.x * orbitSpeed;
                    pitch = Mathf.Clamp(pitch - delta.y * orbitSpeed, pitchLimits.x, pitchLimits.y);
                }
                if (kb.qKey.isPressed) yaw -= keyOrbitSpeed * dt;
                if (kb.eKey.isPressed) yaw += keyOrbitSpeed * dt;

                // --- Zoom ---
                float scroll = mouse.scroll.ReadValue().y;
                if (Mathf.Abs(scroll) > 0.01f)
                {
                    distance = Mathf.Clamp(
                        distance * Mathf.Pow(1.1f, -scroll * zoomSpeed * 0.1f),
                        distanceLimits.x, distanceLimits.y);
                }
            }

            // Track followed creature if active
            if (_followIndex >= 0 && _creatures != null && _followIndex < _creatures.Length && _creatures[_followIndex] != null)
            {
                Vector3 targetPivot = _creatures[_followIndex].transform.position + Vector3.up * 0.85f;
                pivot = Vector3.Lerp(pivot, targetPivot, 12f * dt);
            }
            else if (_world != null)
            {
                float groundY = _world.SampleHeight(pivot.x, pivot.z) + 1.1f;
                pivot.y = Mathf.Lerp(pivot.y, groundY, 10f * dt);
            }

            // --- Apply ---
            var rot = Quaternion.Euler(pitch, yaw, 0f);
            Vector3 desiredCamPos = pivot - rot * Vector3.forward * distance;
            if (_world != null)
            {
                float minCamY = _world.SampleHeight(desiredCamPos.x, desiredCamPos.z) + 0.65f;
                if (desiredCamPos.y < minCamY)
                    desiredCamPos.y = minCamY;
            }
            transform.position = desiredCamPos;
            transform.rotation = Quaternion.LookRotation(pivot - transform.position);
        }

        private void OnGUI()
        {
            if (!_showHud)
                return;

            GUI.color = new Color(0f, 0f, 0f, 0.65f);
            GUI.Box(new Rect(14, 14, 420, _followIndex >= 0 ? 118 : 68), GUIContent.none);
            GUI.color = Color.white;

            GUI.Label(new Rect(24, 20, 400, 22), "<b>aniML World — Track A: 38-DOF PolyArt Deer RL Bridge</b>");
            GUI.Label(new Rect(24, 40, 400, 20), "WASD: Pan | RMB/QE: Orbit | Scroll: Zoom | <b>Tab</b>: Follow Deer | H: HUD");

            if (_followIndex >= 0 && _creatures != null && _followIndex < _creatures.Length && _creatures[_followIndex] != null)
            {
                var c = _creatures[_followIndex];
                var brain = c.GetComponent<DeerEcologyBrain>();
                var motor = c.GetComponent<DeerPolicyLocomotionMotor>();
                string policyName = motor != null ? motor.PolicyResourcePath.Replace("Policies/", "") : "n/a";
                string stateName = brain != null ? brain.State.ToString() : "n/a";
                float speed = c.Motor != null ? c.Motor.CurrentSpeed : 0f;

                GUI.Label(new Rect(24, 64, 400, 20),
                    $"Following: <b>{c.name}</b>  |  Policy: <b>{policyName}</b> (50 Hz ELU)");
                if (brain != null)
                {
                    GUI.Label(new Rect(24, 84, 400, 20),
                        $"State: <b>{stateName}</b>  |  Speed: <b>{speed:F2} m/s</b>  |  Hunger: {brain.Hunger01:P0}  Thirst: {brain.Thirst01:P0}");
                }
            }
        }
    }
}
