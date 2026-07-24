using UnityEngine;

namespace AniML.World
{
    /// <summary>
    /// All parameters of the procedural world. One asset = one world recipe;
    /// the same data drives generation, editing brushes (later), and the sim's
    /// reads (biome lookups). Deterministic per seed.
    /// </summary>
    [CreateAssetMenu(menuName = "AniML/World Settings", fileName = "WorldSettings")]
    public sealed class WorldSettings : ScriptableObject
    {
        [Header("Dimensions")]
        [Tooltip("World is square, worldSize x worldSize meters, centered on origin.")]
        public float WorldSize = 1024f;
        public float ChunkSize = 64f;
        [Tooltip("Meters per terrain quad. Smaller = denser mesh.")]
        public float QuadSize = 2f;

        [Header("Height")]
        public int Seed = 12345;
        public float HeightAmplitude = 90f;
        [Tooltip("Noise feature size in meters (bigger = broader hills).")]
        public float HeightFeatureSize = 300f;
        [Range(1, 8)] public int Octaves = 5;
        [Range(1.5f, 3.5f)] public float Lacunarity = 2.0f;
        [Range(0.2f, 0.8f)] public float Persistence = 0.48f;
        [Tooltip("Exponent > 1 flattens lowlands and sharpens peaks.")]
        public float HeightExponent = 1.7f;
        [Tooltip("Exponent shaping the island falloff toward the edges.")]
        public float IslandFalloffPower = 2.6f;
        [Tooltip("How far below sea level the map edges sink.")]
        public float EdgeSinkDepth = 22f;

        [Header("Mountains (ridged layer, masked to a region)")]
        public float MountainAmplitude = 135f;
        public float MountainFeatureSize = 260f;
        [Tooltip("Feature size of the mask that decides WHERE mountains are.")]
        public float MountainMaskFeatureSize = 750f;
        [Range(0f, 1f)] public float MountainMaskThreshold = 0.47f;

        [Header("Climate")]
        [Tooltip("Meters from water at which moisture has decayed to ~37%.")]
        public float MoistureDecayDistance = 260f;
        [Tooltip("Noise detail blended into the water-distance moisture field.")]
        public float MoistureFeatureSize = 260f;
        [Range(0f, 0.6f)] public float MoistureNoiseAmount = 0.22f;
        [Tooltip("Ambient island humidity added everywhere (sea air).")]
        [Range(0f, 0.5f)] public float BaseMoisture = 0.22f;
        [Tooltip("Prevailing wind (XZ). Rain shadow forms downwind of mountains.")]
        public Vector2 WindDirection = new(1f, 0.25f);
        [Range(0f, 0.6f)] public float RainShadowStrength = 0.35f;
        [Tooltip("North-south temperature difference (0..1 scale). North (+Z) is colder.")]
        [Range(0f, 1f)] public float LatitudeTempRange = 0.45f;
        [Tooltip("Temperature drop per meter of altitude (0..1 scale).")]
        public float AltitudeLapse = 0.0042f;

        [Header("Forests (cluster objects, not per-tree dice rolls)")]
        [Range(0, 24)] public int ForestCount = 11;
        public Vector2 ForestRadiusRange = new(70f, 180f);
        [Tooltip("Density below this is clamped to zero — keeps plains truly open.")]
        [Range(0f, 0.5f)] public float ForestDensityCutoff = 0.16f;

        [Header("Rocky terrain")]
        [Tooltip("Feature size of the mask deciding WHERE badlands/rocky ground appears.")]
        public float RockinessFeatureSize = 480f;
        [Range(0f, 1f)] public float RockinessThreshold = 0.50f;
        [Tooltip("Extra jagged relief amplitude inside rocky zones, meters.")]
        public float RockRoughAmplitude = 10f;
        public float RockRoughFeatureSize = 42f;

        [Header("Domain warp")]
        [Tooltip("Meters of coordinate warp applied to height/moisture/falloff inputs.")]
        public float WarpStrength = 55f;
        public float WarpFeatureSize = 420f;

        [Header("Sea")]
        [Tooltip("World Y of the water surface.")]
        public float SeaLevel = 0f;

        [Header("Rivers")]
        [Range(0, 6)] public int RiverCount = 3;
        public float RiverDepth = 3f;
        [Tooltip("Channel width at the source, meters.")]
        public float RiverMinWidth = 4f;
        [Tooltip("Channel width near the sea, meters.")]
        public float RiverMaxWidth = 11f;

        [Header("Biome palette (height-banded, moisture-tinted)")]
        public Color DeepWaterBed = new Color(0.18f, 0.32f, 0.34f);
        public Color Sand = new Color(0.87f, 0.76f, 0.48f);
        public Color GrassDry = new Color(0.56f, 0.60f, 0.28f);
        public Color GrassLush = new Color(0.33f, 0.58f, 0.24f);
        public Color Forest = new Color(0.26f, 0.48f, 0.22f);
        public Color Rock = new Color(0.52f, 0.48f, 0.44f);
        public Color Snow = new Color(0.94f, 0.95f, 0.97f);
        public Color RiverMud = new Color(0.55f, 0.48f, 0.34f);
        public Color CanyonRockLight = new Color(0.72f, 0.45f, 0.28f);
        public Color CanyonRockDark = new Color(0.52f, 0.29f, 0.20f);
        public Color Scrub = new Color(0.63f, 0.58f, 0.33f);
        public Color Wetland = new Color(0.30f, 0.46f, 0.31f);
        public Color AlpineMeadow = new Color(0.47f, 0.60f, 0.40f);

        [Header("Biome thresholds (meters above sea level)")]
        public float SandBandTop = 2.5f;
        public float GrassBandTop = 34f;
        public float RockBandTop = 62f;
    }
}
