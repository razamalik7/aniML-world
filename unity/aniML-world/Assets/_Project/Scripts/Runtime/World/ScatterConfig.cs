using System;
using System.Collections.Generic;
using UnityEngine;

namespace AniML.World
{
    /// <summary>
    /// One vegetation/prop layer: which prefabs, where they're allowed, and
    /// how dense. Layers are evaluated against the world-truth functions
    /// (height / moisture / slope), so scatter placement is deterministic per
    /// seed and re-derivable by the sim.
    /// </summary>
    [Serializable]
    public sealed class ScatterLayer
    {
        public string Name = "Layer";
        public List<GameObject> Prefabs = new();

        [Header("Placement rules")]
        [Tooltip("Meters above sea level, inclusive band.")]
        public float MinHeight = 3f;
        public float MaxHeight = 30f;
        [Tooltip("Maximum terrain slope in degrees.")]
        public float MaxSlopeDeg = 30f;
        [Tooltip("Moisture gate: probability ramps 0→1 across this range.")]
        public float MoistureStart = 0f;
        public float MoistureFull = 0f;

        [Header("Density")]
        [Tooltip("Expected instances per square meter where all rules pass.")]
        public float DensityPerSqm = 0.01f;
        [Tooltip("Trees: placement follows the forest-object density field instead of the moisture gate.")]
        public bool DriveByForestDensity = false;
        [Tooltip("0..1: how strongly deep forest suppresses this layer (grass thins under canopy).")]
        [Range(0f, 1f)] public float ForestSuppression = 0f;

        [Header("Ecology")]
        [Tooltip("Biomes this species may grow in. Empty = anywhere the other rules allow.")]
        public List<Biome> AllowedBiomes = new();
        [Tooltip("Preferred moisture (0 dry .. 1 wet). Density falls off away from it. -1 = indifferent.")]
        [Range(-1f, 1f)] public float WaterNeed = -1f;
        [Tooltip("Competition radius in meters at scale 1 (0 = exempt from thinning). Larger plants win overlaps.")]
        public float CanopyRadius = 0f;
        [Tooltip("Placement bias toward craggy ground (badlands + outcrops).")]
        public bool PreferRocky = false;
        [Tooltip("Maximum standing-water depth (m) tolerated. 0 = dry ground only; reeds wade.")]
        public float MaxWaterDepth = 0f;
        [Tooltip("Vertical sink (fraction of scale) so big rocks sit IN the ground, not on it.")]
        public float YOffset = 0f;
        [Tooltip("Random tilt in degrees — rocks settle, plants lean.")]
        public float MaxTiltDeg = 0f;
        [Tooltip("This layer is the coastal cliff wall: instances come from the coast tracer, not random sampling.")]
        public bool CoastalWall = false;
        [Tooltip("This layer dresses inland cliff walls of one rock family: instances come from the slope tracer.")]
        public bool InlandWall = false;
        public CliffFamily InlandFamily = CliffFamily.Granite;
        [Tooltip("Temperature band this species tolerates (0 cold .. 1 warm). Overlapping bands blend at borders.")]
        public float MinTemp01 = 0f;
        public float MaxTemp01 = 1f;
        [Tooltip("Random per-instance vertical stretch (±fraction) — same mesh reads taller or squatter.")]
        [Range(0f, 0.4f)] public float YStretch = 0f;

        [Header("Appearance")]
        public Vector2 ScaleRange = new(0.9f, 1.3f);
        [Tooltip("Random offset within the sample cell, as fraction of spacing.")]
        public float PositionJitter = 0.9f;

        [Header("Sim")]
        [Tooltip("Marks instances that the ecosystem treats as food/fuel later.")]
        public bool IsVegetation = true;
    }

    [CreateAssetMenu(menuName = "AniML/Scatter Config", fileName = "ScatterConfig")]
    public sealed class ScatterConfig : ScriptableObject
    {
        public List<ScatterLayer> Layers = new();
    }
}
