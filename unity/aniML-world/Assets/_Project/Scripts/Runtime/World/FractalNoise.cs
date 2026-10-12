using UnityEngine;

namespace AniML.World
{
    /// <summary>
    /// Seeded fractal Brownian motion over Unity's Perlin noise.
    /// Deterministic per seed — required for regenerate-same-world and
    /// future save/load (store seed + edits, not the mesh).
    /// </summary>
    public sealed class FractalNoise
    {
        private readonly Vector2[] _octaveOffsets;
        private readonly float _lacunarity;
        private readonly float _persistence;
        private readonly float _normalization;

        public FractalNoise(int seed, int octaves, float lacunarity, float persistence)
        {
            var rng = new System.Random(seed);
            _octaveOffsets = new Vector2[octaves];
            for (int i = 0; i < octaves; i++)
            {
                _octaveOffsets[i] = new Vector2(
                    (float)(rng.NextDouble() * 2000.0 - 1000.0),
                    (float)(rng.NextDouble() * 2000.0 - 1000.0));
            }

            _lacunarity = lacunarity;
            _persistence = persistence;

            float amp = 1f, total = 0f;
            for (int i = 0; i < octaves; i++) { total += amp; amp *= persistence; }
            _normalization = total;
        }

        /// <summary>Fractal noise in [0, 1]. frequency = 1 / featureSizeMeters.</summary>
        public float Sample(float x, float z, float frequency)
        {
            float value = 0f, amp = 1f, freq = frequency;
            for (int i = 0; i < _octaveOffsets.Length; i++)
            {
                var o = _octaveOffsets[i];
                value += Mathf.PerlinNoise(x * freq + o.x, z * freq + o.y) * amp;
                amp *= _persistence;
                freq *= _lacunarity;
            }
            return value / _normalization;
        }
    }
}
