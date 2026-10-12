using UnityEngine;

namespace AniML.World
{
    /// <summary>
    /// One set of fire physics for both the genesis burn (worldgen runs a
    /// real fire once, so the old scar has real provenance) and the live
    /// wildfire system. Fuel and dryness come from world truth; wind favors
    /// downwind spread.
    /// </summary>
    public static class FireRules
    {
        public static float Fuel(ProceduralWorld w, float x, float z)
        {
            if (w.StandingWaterDepth(x, z) > 0.05f)
                return 0f;
            var b = w.SampleBiome(x, z);
            if (b == Biome.Ocean || b == Biome.Beach || b == Biome.RiverChannel
                || b == Biome.Snow || b == Biome.Rock)
                return 0f;
            float f = w.ForestDensity01(x, z) * 1.2f;
            if (b == Biome.Meadow) f += 0.3f;
            if (b == Biome.Scrub) f += 0.5f;
            if (b == Biome.Wetland) f += 0.08f;
            return Mathf.Clamp01(f);
        }

        public static float Dryness(ProceduralWorld w, float x, float z, float wetness)
            => Mathf.Clamp01((1f - w.SampleMoisture(x, z)) * (1f - wetness * 0.9f) + 0.15f);

        public static float SpreadChance(float fuel, float dryness, float windDot)
            => Mathf.Clamp01(0.42f * fuel * dryness * (0.55f + 0.85f * Mathf.Max(0f, windDot)));
    }
}
