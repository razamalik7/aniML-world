using UnityEngine;

namespace AniML.World
{
    /// <summary>
    /// When the main camera dips below sea level, the world reads as
    /// underwater: dense blue-green fog and cooler ambience. Restores the
    /// surface settings the moment the camera comes back up.
    /// </summary>
    public sealed class UnderwaterTint : MonoBehaviour
    {
        private ProceduralWorld _world;
        private bool _under;
        private float _surfaceDensity;
        private Color _surfaceColor;

        private static readonly Color WaterFog = new(0.10f, 0.32f, 0.38f);

        private void Start()
        {
            _world = GetComponent<ProceduralWorld>();
        }

        private void Update()
        {
            if (!Application.isPlaying || _world == null || _world.Settings == null)
                return;
            var cam = Camera.main;
            if (cam == null)
                return;

            bool nowUnder = cam.transform.position.y < _world.Settings.SeaLevel - 0.05f;
            if (nowUnder && !_under)
            {
                _surfaceDensity = RenderSettings.fogDensity;
                _surfaceColor = RenderSettings.fogColor;
                RenderSettings.fogDensity = 0.055f;
                RenderSettings.fogColor = WaterFog;
                _under = true;
            }
            else if (!nowUnder && _under)
            {
                RenderSettings.fogDensity = _surfaceDensity;
                RenderSettings.fogColor = _surfaceColor;
                _under = false;
            }
        }
    }
}
