using UnityEngine;

namespace AniML.World
{
    /// <summary>
    /// Day/night: drives the sun (directional light), ambient trilight, fog
    /// color, and the Synty skydome material through a 24-hour cycle. Time
    /// advances in Play mode; captures and the editor can pin TimeOfDay.
    /// Built by us (systems are ours); the skydome art is Synty's.
    /// </summary>
    [ExecuteAlways]
    public sealed class SkySystem : MonoBehaviour
    {
        [Range(0f, 24f)] public float TimeOfDay = 10.5f;
        [Tooltip("Game hours per real second in Play mode. 0 pauses the sun.")]
        public float HoursPerSecond = 0.02f;
        [Tooltip("Compass direction of sunrise, degrees.")]
        public float SunAzimuth = 65f;

        [SerializeField] private Light sun;
        private Light _moon;   // separate light so the procedural skybox
                               // still sees the real (set) sun and darkens

        private static readonly Color DayAmbientSky = new(0.62f, 0.72f, 0.85f);
        private static readonly Color DayAmbientEquator = new(0.55f, 0.58f, 0.52f);
        private static readonly Color DayAmbientGround = new(0.32f, 0.30f, 0.26f);
        private static readonly Color NightAmbientSky = new(0.17f, 0.21f, 0.33f);
        private static readonly Color NightAmbientEquator = new(0.12f, 0.15f, 0.24f);
        private static readonly Color NightAmbientGround = new(0.07f, 0.08f, 0.13f);
        private static readonly Color SunDay = new(1.0f, 0.96f, 0.88f);
        private static readonly Color SunLow = new(1.0f, 0.62f, 0.36f);
        private static readonly Color MoonColor = new(0.42f, 0.52f, 0.72f);

        public void Configure(Light sunLight)
        {
            sun = sunLight;
        }

        private void Update()
        {
            if (Application.isPlaying && HoursPerSecond > 0f)
            {
                TimeOfDay = (TimeOfDay + Time.deltaTime * HoursPerSecond) % 24f;
                Apply();
            }
            else if (!Application.isPlaying)
            {
                Apply();
            }
        }

        /// <summary>Push the current TimeOfDay into lights and ambience.</summary>
        public void Apply()
        {
            if (sun == null)
                return;

            // The procedural skybox must track the REAL sun (else it grabs
            // the brightest light — the moon — and stays daytime-blue).
            RenderSettings.sun = sun;

            // Sun elevation: -90 at midnight, +90 at noon-ish.
            float dayT = TimeOfDay / 24f;
            float elevation = Mathf.Sin((dayT - 0.25f) * Mathf.PI * 2f) * 78f;
            sun.transform.rotation = Quaternion.Euler(elevation, SunAzimuth, 0f);

            // Day strength: 1 at high noon, 0 below the horizon.
            float day = Mathf.Clamp01(Mathf.InverseLerp(-6f, 18f, elevation));
            // Warmth: strongest when the sun is low but up (sunrise/sunset).
            float low = Mathf.Clamp01(Mathf.InverseLerp(24f, 2f, elevation))
                        * Mathf.Clamp01(Mathf.InverseLerp(-8f, 4f, elevation));

            sun.color = Color.Lerp(SunDay, SunLow, low);
            sun.intensity = Mathf.Lerp(0f, 1.15f, day);

            // A dedicated shadowless moon carries the night so the ground
            // stays readable while the skybox (which follows the sun) darkens.
            if (_moon == null)
            {
                var go = new GameObject("Moon") { hideFlags = HideFlags.DontSave };
                go.transform.SetParent(transform, false);
                _moon = go.AddComponent<Light>();
                _moon.type = LightType.Directional;
                _moon.shadows = LightShadows.None;
            }
            _moon.transform.rotation = Quaternion.Euler(42f, SunAzimuth + 180f, 0f);
            _moon.color = MoonColor;
            _moon.intensity = Mathf.Lerp(0.34f, 0f, day);

            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = Color.Lerp(NightAmbientSky, DayAmbientSky, day);
            RenderSettings.ambientEquatorColor = Color.Lerp(NightAmbientEquator, DayAmbientEquator, day);
            RenderSettings.ambientGroundColor = Color.Lerp(NightAmbientGround, DayAmbientGround, day);

            if (RenderSettings.fog)
            {
                RenderSettings.fogColor = Color.Lerp(
                    new Color(0.08f, 0.10f, 0.16f),
                    Color.Lerp(new Color(0.75f, 0.82f, 0.92f), new Color(0.95f, 0.68f, 0.45f), low),
                    day);
            }
        }
    }
}
