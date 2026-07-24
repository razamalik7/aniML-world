using UnityEngine;
using UnityEngine.InputSystem;

namespace AniML.CameraRig
{
    /// <summary>
    /// RTS-style sandbox camera: WASD pan, scroll zoom, hold-right-mouse (or Q/E) orbit,
    /// middle-mouse drag pan. Operates on a pivot point; the camera looks at the pivot
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
        [SerializeField] private Vector2 pitchLimits = new Vector2(15f, 80f);

        [Header("Zoom")]
        [SerializeField] private float zoomSpeed = 3f;
        [SerializeField] private Vector2 distanceLimits = new Vector2(8f, 150f);

        [Header("State")]
        [SerializeField] private Vector3 pivot = Vector3.zero;
        [SerializeField] private float yaw = 45f;
        [SerializeField] private float pitch = 45f;
        [SerializeField] private float distance = 40f;

        private void LateUpdate()
        {
            var kb = Keyboard.current;
            var mouse = Mouse.current;
            if (kb == null || mouse == null)
                return;

            float dt = Time.unscaledDeltaTime;

            // --- Keyboard pan (relative to camera yaw, on the ground plane) ---
            var panInput = Vector2.zero;
            if (kb.wKey.isPressed) panInput.y += 1f;
            if (kb.sKey.isPressed) panInput.y -= 1f;
            if (kb.dKey.isPressed) panInput.x += 1f;
            if (kb.aKey.isPressed) panInput.x -= 1f;

            var yawRot = Quaternion.Euler(0f, yaw, 0f);
            // Scale pan with distance so travel feels constant at any zoom.
            float panScale = panSpeed * Mathf.Max(distance / 40f, 0.3f);
            pivot += yawRot * new Vector3(panInput.x, 0f, panInput.y) * (panScale * dt);

            // --- Middle-mouse drag pan ---
            if (mouse.middleButton.isPressed)
            {
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

            // --- Apply ---
            var rot = Quaternion.Euler(pitch, yaw, 0f);
            transform.position = pivot - rot * Vector3.forward * distance;
            transform.rotation = rot;
        }
    }
}
