using System.Collections.Generic;
using UnityEngine;

namespace AniML.World
{
    /// <summary>
    /// Generates the faceted low-poly terrain from WorldSettings: chunked
    /// meshes with per-face biome colors and mesh colliders. The height and
    /// biome functions here are THE world truth — the scatter system, sim,
    /// and editing brushes all read the same functions.
    /// </summary>
    public sealed class ProceduralWorld : MonoBehaviour
    {
        [SerializeField] private WorldSettings settings;
        [SerializeField] private Material terrainMaterial;
        [SerializeField] private Material waterMaterial;
        [SerializeField] private Material riverWaterMaterial;
        [SerializeField] private Material lakeWaterMaterial;

        private FractalNoise _heightNoise;
        private FractalNoise _moistureNoise;
        private FractalNoise _mountainNoise;
        private FractalNoise _mountainMaskNoise;
        private FractalNoise _warpNoiseX;
        private FractalNoise _warpNoiseZ;

        // Moisture: baked distance-from-water field (coarse grid, bilinear).
        private float[,] _moistureGrid;
        private float[,] _oceanDistGrid;
        private const float MoistureCellSize = 8f;
        private int _moistureRes;

        // Rock outcrops: crag objects, like forests but stone.
        private readonly List<ForestSeed> _outcropSeeds = new();

        // Ponds: explicit shallow basins with a water level (lakes-lite until
        // the Wave 2 priority-flood grid arrives).
        public struct PondStamp { public Vector2 Pos; public float Radius; public float BottomY; public float WaterY; }
        private readonly List<PondStamp> _ponds = new();
        public IReadOnlyList<PondStamp> Ponds => _ponds;

        // Lakes: depressions found by priority-flood on the baked grid.
        public sealed class Lake
        {
            public float WaterLevel;
            public readonly List<(int gx, int gz)> Cells = new();
        }
        private readonly List<Lake> _lakes = new();
        private readonly Dictionary<long, int> _lakeCellIndex = new();
        public IReadOnlyList<Lake> Lakes => _lakes;

        // Coastal cliff walls: aligned segments along steep coastline runs.
        public struct CoastCliff { public Vector3 Pos; public float YawDeg; public float Scale; }
        private readonly List<CoastCliff> _coastCliffs = new();
        public IReadOnlyList<CoastCliff> CoastCliffs => _coastCliffs;

        // Inland cliffs: chained wall meshes dressing steep terrain anywhere,
        // so slopes read as built geology instead of stretched texture.
        public struct InlandCliff { public Vector3 Pos; public float YawDeg; public float Scale; public CliffFamily Family; }
        private readonly List<InlandCliff> _inlandCliffs = new();
        public IReadOnlyList<InlandCliff> InlandCliffs => _inlandCliffs;

        // Summits: the tallest peaks carry scaled MountainRange silhouettes.
        public struct SummitSite { public Vector3 Pos; public float YawDeg; public float Scale; public int Variant; }
        private readonly List<SummitSite> _summits = new();
        public IReadOnlyList<SummitSite> Summits => _summits;

        // Which river carries THE canyon (-1 = none).
        private int _gorgeRiver = -1;

        // The world's fault line (tectonic scarp), seeded per world.
        private Vector2 _faultPoint, _faultDir, _faultNormal;
        private FractalNoise _faultNoise;
        private FractalNoise _duneNoise;

        // Every dial a LandformProfile moves, resolved once per generation.
        private struct ProfileDials
        {
            public float BaseMoisture, RainShadow, MoistureDecay, FreshGain;
            public float MountainAmpMul, MaskThresholdDelta;
            public float TempOffset;
            public float CanyonDepth;
            public int SummitCount;
            public float GraniteAltitude;
            public bool MesaTerraces;
            public int RiverCount, PondCount;
            public float MinLakeDepth;
            public int ForestCount;
            public float CliffSlopeOn;   // slope (deg) that seeds a cliff wall
            public float WetlandReach;   // how far marsh spreads from fresh water
            public bool Volcano;
            public float HeightMul;      // overall relief multiplier
            public bool GlacialCarve;    // U-valleys + moraines above the paleo-snowline
            public float PaleoSnowline;  // meters above sea where the old ice began
            public bool Dunes;           // wind-oriented dune fields on dry flats
            public float FaultScarpH;    // tectonic step height (0 = no fault)
            public int KarstCount;       // sinkhole bowls (deep-broad ones flood)
        }
        private ProfileDials _dials;

        private void ResolveProfileDials()
        {
            switch (settings.Profile)
            {
                case LandformProfile.AlpineGranite:
                    _dials = new ProfileDials
                    {
                        BaseMoisture = settings.BaseMoisture + 0.04f,
                        RainShadow = settings.RainShadowStrength,
                        MoistureDecay = settings.MoistureDecayDistance,
                        FreshGain = 0.62f,
                        MountainAmpMul = 1.25f,
                        MaskThresholdDelta = -0.04f,   // more of the island is range
                        TempOffset = -0.14f,           // snowline drops
                        CanyonDepth = 22f,
                        SummitCount = 3,
                        GraniteAltitude = 34f,
                        HeightMul = 1f,
                        WetlandReach = 55f,
                        GlacialCarve = true,
                        PaleoSnowline = 26f,
                        FaultScarpH = 7f,
                        KarstCount = 1,
                        MesaTerraces = false,
                        RiverCount = settings.RiverCount,
                        PondCount = 4,
                        MinLakeDepth = 0.9f,
                        ForestCount = settings.ForestCount,
                        // With this much steep ground, dress only the truly
                        // sheer — crags at crests, not full rock armor.
                        CliffSlopeOn = 47f,
                    };
                    break;

                case LandformProfile.Canyonlands:
                    _dials = new ProfileDials
                    {
                        // A desert is dry because water is SCARCE, not just
                        // distant: one river, no ponds, weak freshwater reach.
                        BaseMoisture = 0.10f,
                        RainShadow = 0.55f,
                        MoistureDecay = 130f,
                        FreshGain = 0.30f,
                        MountainAmpMul = 0.55f,        // mesas, not peaks
                        MaskThresholdDelta = 0.06f,
                        TempOffset = 0.18f,            // no snow anywhere
                        CanyonDepth = 38f,
                        SummitCount = 0,               // no alpine ranges here
                        GraniteAltitude = 999f,        // red rock owns the heights
                        HeightMul = 1f,
                        WetlandReach = 55f,
                        MesaTerraces = true,
                        RiverCount = 1,                // THE canyon's river, alone
                        PondCount = 0,
                        MinLakeDepth = 4.5f,           // shallow basins stay dry playas;
                                                       // only true deeps become oases
                        ForestCount = 3,               // cottonwood stands, not woodland
                        CliffSlopeOn = 40f,
                        Dunes = true,
                        FaultScarpH = 11f,
                    };
                    break;

                case LandformProfile.SwampDelta:
                    _dials = new ProfileDials
                    {
                        // Everything low, warm, and waterlogged — the wetland
                        // classifier does the rest and the mangroves take over.
                        BaseMoisture = 0.5f,
                        RainShadow = 0.1f,
                        MoistureDecay = 400f,
                        FreshGain = 0.7f,
                        MountainAmpMul = 0.35f,
                        MaskThresholdDelta = 0.12f,
                        TempOffset = 0.12f,
                        CanyonDepth = 8f,
                        SummitCount = 0,
                        GraniteAltitude = 999f,
                        MesaTerraces = false,
                        RiverCount = 5,           // a braid of channels
                        PondCount = 6,
                        MinLakeDepth = 0.5f,      // standing water everywhere
                        ForestCount = 10,
                        CliffSlopeOn = 45f,
                        Volcano = false,
                        HeightMul = 0.6f,
                        WetlandReach = 170f,   // the marsh owns the lowlands
                        KarstCount = 2,
                    };
                    break;

                case LandformProfile.VolcanoIsle:
                    _dials = new ProfileDials
                    {
                        // Lush tropical lowlands; the drama is the cone.
                        BaseMoisture = settings.BaseMoisture + 0.02f,
                        RainShadow = 0.3f,
                        MoistureDecay = 220f,
                        FreshGain = 0.55f,
                        MountainAmpMul = 0.7f,
                        MaskThresholdDelta = 0.1f,     // noise ranges yield to the stamp
                        TempOffset = 0.22f,            // tropical, no snow
                        CanyonDepth = 14f,
                        SummitCount = 0,               // the cone IS the summit
                        GraniteAltitude = 20f,         // dark basalt crags up high
                        HeightMul = 1f,
                        WetlandReach = 55f,
                        MesaTerraces = false,
                        RiverCount = 2,
                        PondCount = 2,
                        MinLakeDepth = 0.9f,
                        ForestCount = 8,
                        CliffSlopeOn = 42f,
                        Volcano = true,
                    };
                    break;

                default: // TemperateIsles — the tuned baseline, unchanged.
                    _dials = new ProfileDials
                    {
                        BaseMoisture = settings.BaseMoisture,
                        RainShadow = settings.RainShadowStrength,
                        MoistureDecay = settings.MoistureDecayDistance,
                        FreshGain = 0.62f,
                        MountainAmpMul = 1f,
                        MaskThresholdDelta = 0f,
                        TempOffset = 0f,
                        CanyonDepth = 30f,
                        SummitCount = 2,
                        GraniteAltitude = 46f,
                        HeightMul = 1f,
                        WetlandReach = 55f,
                        MesaTerraces = false,
                        RiverCount = settings.RiverCount,
                        PondCount = 4,
                        MinLakeDepth = 0.9f,
                        ForestCount = settings.ForestCount,
                        CliffSlopeOn = 40f,
                        FaultScarpH = 5f,
                        KarstCount = 3,
                    };
                    break;
            }

            // Recipe variance: every seed rolls its own inventory inside the
            // profile's character — one seed gets a faultless island with
            // four rivers, another a mild gorge and a double karst country.
            // (Same seed still always rolls the same recipe.)
            if (settings.RecipeVariance)
            {
                var vr = new System.Random(settings.Seed * 977 + 13);
                int Jit(int v, int spread, int min) =>
                    v <= 0 ? v : Mathf.Max(min, v + vr.Next(-spread, spread + 1));
                float JitF(float v, float lo, float hi) =>
                    v * Mathf.Lerp(lo, hi, (float)vr.NextDouble());

                _dials.RiverCount = Jit(_dials.RiverCount, 1, 1);
                _dials.PondCount = Jit(_dials.PondCount, 2, 0);
                _dials.ForestCount = Jit(_dials.ForestCount, 3, 1);
                _dials.KarstCount = Jit(_dials.KarstCount, 2, 0);
                _dials.SummitCount = Jit(_dials.SummitCount, 1, 1);
                _dials.CanyonDepth = JitF(_dials.CanyonDepth, 0.6f, 1.4f);
                _dials.MountainAmpMul = JitF(_dials.MountainAmpMul, 0.88f, 1.14f);
                // Roughly one seed in four has no fault at all.
                _dials.FaultScarpH = vr.NextDouble() < 0.25
                    ? 0f
                    : JitF(_dials.FaultScarpH, 0.6f, 1.6f);
            }
        }

        private float[,] _freshDistGrid;

        // Natural wonders: singular one-per-world features at computed sites.
        public struct WonderSite { public string Kind; public Vector3 Pos; public float YawDeg; public float Scale; }
        private readonly List<WonderSite> _wonders = new();
        public IReadOnlyList<WonderSite> Wonders => _wonders;

        // Baked heightfield: analytic height (with river/pond carves) is
        // rasterized once per generation, erosion runs on the grid, and all
        // queries become bilinear samples — faster AND erodible.
        private float[,] _heightGrid;
        private float[,] _sedimentOutflow;
        private int _heightRes;
        private float _heightCell;
        private bool _gridReady;

        // Forests as objects: seeds with radius/strength, minus clearings.
        private struct ForestSeed { public Vector2 Pos; public float Radius; public float Strength; }
        private readonly List<ForestSeed> _forestSeeds = new();
        private readonly List<ForestSeed> _forestClearings = new();

        // Rivers: polylines of (x, y=unused, z, w=width), spatially hashed.
        private readonly List<List<Vector4>> _rivers = new();
        private readonly Dictionary<long, List<Vector4>> _riverGrid = new();
        private const float RiverCellSize = 16f;

        // Connectivity: ford stamps keep rivers wadeable, and the canyon
        // gets one talus-ramp crossing cut through both rims.
        public struct FordSite { public Vector2 Pos; public float Radius; }
        private readonly List<FordSite> _fords = new();
        public IReadOnlyList<FordSite> Fords => _fords;

        private struct GorgeCrossing
        {
            public bool Active;
            public Vector2 Pos, Dir, RiverDir;   // Dir points toward the rims
            public float BedY, RimA, RimB, HalfLen, HalfWidth;
        }
        private GorgeCrossing _gorgeCross;
        public Vector2? CanyonCrossing => _gorgeCross.Active ? _gorgeCross.Pos : (Vector2?)null;
        /// <summary>Unit direction across the crossing, from rim A toward rim B.</summary>
        public Vector2? CanyonCrossingDir => _gorgeCross.Active ? _gorgeCross.Dir : (Vector2?)null;

        /// <summary>Is this point inside the canyon-crossing ramp corridor (plus padding)?</summary>
        public bool InCanyonCrossing(float x, float z, float pad = 0f)
        {
            if (!_gorgeCross.Active)
                return false;
            float dx = x - _gorgeCross.Pos.x, dz = z - _gorgeCross.Pos.y;
            float u = dx * _gorgeCross.Dir.x + dz * _gorgeCross.Dir.y;
            float v = dx * _gorgeCross.RiverDir.x + dz * _gorgeCross.RiverDir.y;
            return Mathf.Abs(u) < _gorgeCross.HalfLen + pad
                   && Mathf.Abs(v) < _gorgeCross.HalfWidth * 2.2f + pad;
        }

        public WorldSettings Settings => settings;

        public void Configure(WorldSettings s, Material terrainMat, Material waterMat,
            Material riverMat = null, Material lakeMat = null)
        {
            settings = s;
            terrainMaterial = terrainMat;
            waterMaterial = waterMat;
            riverWaterMaterial = riverMat;
            lakeWaterMaterial = lakeMat;
        }

        /// <summary>
        /// Visit every river bed drop of at least minDrop meters: position,
        /// downstream yaw (deg), and the drop height. Feeds cascade FX.
        /// </summary>
        public void ForEachRiverDrop(float minDrop, System.Action<Vector3, float, float> visit)
        {
            foreach (var river in _rivers)
            {
                for (int i = 0; i + 1 < river.Count; i++)
                {
                    float drop = river[i].y - river[i + 1].y;
                    if (drop < minDrop)
                        continue;
                    var dir = new Vector2(river[i + 1].x - river[i].x, river[i + 1].z - river[i].z);
                    visit(new Vector3(
                            (river[i].x + river[i + 1].x) * 0.5f,
                            river[i + 1].y + settings.RiverDepth * 0.55f,
                            (river[i].z + river[i + 1].z) * 0.5f),
                        Mathf.Atan2(dir.x, dir.y) * Mathf.Rad2Deg,
                        drop);
                }
            }
        }

        private void Awake()
        {
            // Generated geometry is deliberately NOT serialized (a saved
            // scene with 1.5M terrain vertices is a 134 MB file). The world
            // rebuilds deterministically from the seed at load.
            if (Application.isPlaying && transform.childCount == 0)
                Generate();
        }

        /// <summary>
        /// Initialize the noise samplers without rebuilding meshes — used by
        /// consumers (scatter, sim) when terrain geometry already exists.
        /// </summary>
        public void EnsureSamplers()
        {
            if (_heightNoise != null || settings == null)
                return;
            _gridReady = false;
            ResolveProfileDials();
            ResolveLandmass();

            // One fault line per world: a wandering tectonic step whose
            // trace and throw the profile controls.
            var faultRng = new System.Random(settings.Seed * 211 + 5);
            float faultAng = (float)faultRng.NextDouble() * Mathf.PI;
            _faultDir = new Vector2(Mathf.Cos(faultAng), Mathf.Sin(faultAng));
            _faultNormal = new Vector2(-_faultDir.y, _faultDir.x);
            _faultPoint = new Vector2(
                ((float)faultRng.NextDouble() - 0.5f) * settings.WorldSize * 0.4f,
                ((float)faultRng.NextDouble() - 0.5f) * settings.WorldSize * 0.4f);
            _faultNoise = new FractalNoise(settings.Seed + 424243, 2, 2f, 0.5f);
            _duneNoise = new FractalNoise(settings.Seed + 96557, 2, 2f, 0.5f);

            _heightNoise = new FractalNoise(settings.Seed, settings.Octaves, settings.Lacunarity, settings.Persistence);
            _moistureNoise = new FractalNoise(settings.Seed + 7919, 3, 2.1f, 0.5f);
            _mountainNoise = new FractalNoise(settings.Seed + 104729, 4, 2.2f, 0.5f);
            _mountainMaskNoise = new FractalNoise(settings.Seed + 1299709, 2, 2.0f, 0.5f);
            _warpNoiseX = new FractalNoise(settings.Seed + 15485863, 3, 2.0f, 0.5f);
            _warpNoiseZ = new FractalNoise(settings.Seed + 32452843, 3, 2.0f, 0.5f);
            BuildRivers();
            PlacePonds();
            BakeHeightGrid();
            DetectLakes();
            BakeMoistureGrid();
            ComputeSummits();      // raises summit pedestals in the grid,
            ComputeVolcano();      // (or the cone does) so everything after
            PlaceForests();        // sees them
            ComputeBurnScar();
            PlaceOutcrops();
            TraceCoastCliffs();
            TraceInlandCliffs();
            ComputeWonders();
            ReportWalkability();
        }

        /// <summary>
        /// Generation-time connectivity audit: flood-fill the land on a 4 m
        /// grid (passable = slope ≤ 38°, water ≤ 0.5 m deep) and log how
        /// much of it hangs together. Pure diagnostics — no terrain writes.
        /// </summary>
        private void ReportWalkability()
        {
            const int stride = 2;                       // 2 × 2 m quads = 4 m cells
            float cellD = _heightCell * stride;
            float maxGrad = Mathf.Tan(38f * Mathf.Deg2Rad) * cellD;
            float half = settings.WorldSize * 0.5f;
            int n = (_heightRes - 1) / stride + 1;

            bool[,] passable = new bool[n, n];
            int landCells = 0, passCells = 0;
            for (int cz = 0; cz < n; cz++)
            {
                for (int cx = 0; cx < n; cx++)
                {
                    int gx = Mathf.Min(cx * stride, _heightRes - 1);
                    int gz = Mathf.Min(cz * stride, _heightRes - 1);
                    float h = _heightGrid[gx, gz];
                    if (h <= settings.SeaLevel + 0.05f)
                        continue;
                    landCells++;

                    float x = gx * _heightCell - half, z = gz * _heightCell - half;
                    if (StandingWaterDepth(x, z) > 0.5f)
                        continue;
                    GetRiverData(x, z, out float rinf, out float rbed);
                    if (rinf > 0.1f && rbed + settings.RiverDepth * 0.55f - h > 0.5f)
                        continue;

                    float rise = 0f;
                    if (gx + stride < _heightRes)
                        rise = Mathf.Max(rise, Mathf.Abs(_heightGrid[gx + stride, gz] - h));
                    if (gz + stride < _heightRes)
                        rise = Mathf.Max(rise, Mathf.Abs(_heightGrid[gx, gz + stride] - h));
                    if (rise > maxGrad)
                        continue;

                    passable[cx, cz] = true;
                    passCells++;
                }
            }

            // BFS components over 4-connectivity.
            int[,] label = new int[n, n];
            var queue = new Queue<Vector2Int>();
            int regions = 0, bigRegions = 0, largest = 0;
            int haCells = Mathf.RoundToInt(10000f / (cellD * cellD));   // cells per hectare
            for (int sz = 0; sz < n; sz++)
            {
                for (int sx = 0; sx < n; sx++)
                {
                    if (!passable[sx, sz] || label[sx, sz] != 0)
                        continue;
                    regions++;
                    int size = 0;
                    label[sx, sz] = regions;
                    queue.Enqueue(new Vector2Int(sx, sz));
                    while (queue.Count > 0)
                    {
                        var c = queue.Dequeue();
                        size++;
                        for (int d = 0; d < 4; d++)
                        {
                            int nx = c.x + (d == 0 ? 1 : d == 1 ? -1 : 0);
                            int nz = c.y + (d == 2 ? 1 : d == 3 ? -1 : 0);
                            if (nx < 0 || nz < 0 || nx >= n || nz >= n)
                                continue;
                            if (!passable[nx, nz] || label[nx, nz] != 0)
                                continue;
                            label[nx, nz] = regions;
                            queue.Enqueue(new Vector2Int(nx, nz));
                        }
                    }
                    if (size >= haCells)
                        bigRegions++;
                    largest = Mathf.Max(largest, size);
                }
            }

            float passPct = landCells > 0 ? passCells * 100f / landCells : 0f;
            float mainPct = passCells > 0 ? largest * 100f / passCells : 0f;
            string crossing = _gorgeCross.Active
                ? $"yes at ({_gorgeCross.Pos.x:F0}, {_gorgeCross.Pos.y:F0})"
                : "none";
            Debug.Log($"[AniML] Walkability: {passPct:F1}% of land passable, "
                + $"{mainPct:F1}% of that in the main region "
                + $"({bigRegions} regions >= 1 ha); {_fords.Count} fords; "
                + $"canyon crossing {crossing}.");
        }

        /// <summary>
        /// One of each, at sites the terrain and hydrology single out:
        /// THE waterfall, THE giant tree, THE hot springs, THE whale bones.
        /// Singularity is the point — landmarks you remember and navigate by.
        /// </summary>
        private void ComputeWonders()
        {
            _wonders.Clear();
            var rng = new System.Random(settings.Seed * 71 + 19);

            // Optional wonders are chance-gated per seed when recipe
            // variance is on — not every island got every miracle.
            var recipeRng = new System.Random(settings.Seed * 397 + 71);
            bool Roll(float p) => !settings.RecipeVariance || recipeRng.NextDouble() < p;

            // Waterfall: the largest single bed drop on any river — on dry
            // land (flow-derived rivers can put the biggest drop at the
            // coastal shelf, and falls pouring into open sea read as a
            // glitch), and preferably in the visitable green band rather
            // than the crag-walled snowfields where nobody can stand.
            Vector3 fallPos = default; float fallYaw = 0f; float fallScale = 1f;
            bool found = false;
            bool PickFalls(float ceilingY)
            {
                float bestDrop = 3.2f;
                bool ok = false;
                foreach (var river in _rivers)
                {
                    for (int i = 0; i + 1 < river.Count; i++)
                    {
                        float drop = river[i].y - river[i + 1].y;
                        if (river[i + 1].y < settings.SeaLevel + 4f || river[i + 1].y > ceilingY)
                            continue;
                        if (drop > bestDrop)
                        {
                            bestDrop = drop;
                            var dir = new Vector2(river[i + 1].x - river[i].x, river[i + 1].z - river[i].z);
                            fallYaw = Mathf.Atan2(dir.x, dir.y) * Mathf.Rad2Deg;
                            fallPos = new Vector3(
                                (river[i].x + river[i + 1].x) * 0.5f,
                                river[i + 1].y + settings.RiverDepth * 0.55f,
                                (river[i].z + river[i + 1].z) * 0.5f);
                            fallScale = Mathf.Clamp(drop / 4f, 1f, 2.4f);
                            ok = true;
                        }
                    }
                }
                return ok;
            }
            found = PickFalls(settings.SeaLevel + 45f) || PickFalls(float.MaxValue);
            if (found)
                _wonders.Add(new WonderSite { Kind = "Waterfall", Pos = fallPos, YawDeg = fallYaw, Scale = fallScale });

            // Giant tree: the clearing of the largest forest.
            if (_forestSeeds.Count > 0)
            {
                var biggest = _forestSeeds[0];
                foreach (var f in _forestSeeds)
                    if (f.Radius > biggest.Radius) biggest = f;
                Vector2 site = biggest.Pos;
                foreach (var c in _forestClearings)
                {
                    if ((c.Pos - biggest.Pos).magnitude < biggest.Radius)
                    { site = c.Pos; break; }
                }
                _wonders.Add(new WonderSite
                {
                    Kind = "GiantTree",
                    Pos = new Vector3(site.x, SampleHeight(site.x, site.y), site.y),
                    YawDeg = (float)rng.NextDouble() * 360f,
                    Scale = 3.4f,
                });
            }

            // Hot springs: craggy scrub, relaxing to any dry/rocky ground.
            if (Roll(0.8f))
            if (!TryPlaceWonder(rng, "HotSprings", 240,
                    (x, z) => SampleBiome(x, z) == Biome.Scrub && SampleCragginess01(x, z) > 0.2f))
            {
                if (!TryPlaceWonder(rng, "HotSprings", 240, (x, z) => SampleBiome(x, z) == Biome.Scrub))
                    TryPlaceWonder(rng, "HotSprings", 240,
                        (x, z) => SampleCragginess01(x, z) > 0.4f && SampleBiome(x, z) != Biome.Ocean
                                  && SampleBiome(x, z) != Biome.Snow);
            }
            if (Roll(0.85f) && TryPlaceWonder(rng, "WhaleBones", 240,
                    (x, z) => SampleBiome(x, z) == Biome.Beach && SampleOceanDistance(x, z) < 24f))
            {
                // A beached whale lies ALONG the waterline, nudged up the
                // sand — not nose-in perpendicular with its tail in the surf.
                var w = _wonders[_wonders.Count - 1];
                const float e = 6f;
                var inland = new Vector2(
                    SampleOceanDistance(w.Pos.x + e, w.Pos.z) - SampleOceanDistance(w.Pos.x - e, w.Pos.z),
                    SampleOceanDistance(w.Pos.x, w.Pos.z + e) - SampleOceanDistance(w.Pos.x, w.Pos.z - e));
                if (inland.sqrMagnitude > 1e-4f)
                {
                    inland.Normalize();
                    w.Pos += new Vector3(inland.x, 0f, inland.y) * 5f;
                    w.Pos.y = SampleHeight(w.Pos.x, w.Pos.z);
                    w.YawDeg = Mathf.Atan2(-inland.y, inland.x) * Mathf.Rad2Deg;
                    _wonders[_wonders.Count - 1] = w;
                }
            }

            // THE red arch: on the canyon rim, spanning parallel to the drop
            // so you can stand under it and look down the slot.
            if (_gorgeRiver >= 0)
            {
                var path = _rivers[_gorgeRiver];
                int idx = Mathf.Clamp(Mathf.RoundToInt(path.Count * 0.5f), 0, path.Count - 2);
                var p = path[idx];
                var next = path[idx + 1];
                var dir = new Vector2(next.x - p.x, next.z - p.z).normalized;
                var side = new Vector2(-dir.y, dir.x);
                float off = p.w + 9f;
                var rim = new Vector2(p.x + side.x * off, p.z + side.y * off);
                _wonders.Add(new WonderSite
                {
                    Kind = "RedArch",
                    Pos = new Vector3(rim.x, SampleHeight(rim.x, rim.y), rim.y),
                    YawDeg = Mathf.Atan2(dir.x, dir.y) * Mathf.Rad2Deg,
                    Scale = 1.3f,
                });
                // The arch stands ON the rim wall line — clear the rim crag
                // pieces around it so it reads as the feature, not one rock
                // among many.
                _inlandCliffs.RemoveAll(c =>
                    (c.Pos.x - rim.x) * (c.Pos.x - rim.x)
                    + (c.Pos.z - rim.y) * (c.Pos.z - rim.y) < 20f * 20f);
            }

            // THE ice arch: out on the high snowfield (the Alpine "arch" is
            // honestly glacier ice — so it lives where ice belongs).
            if (Roll(0.8f))
                TryPlaceWonder(rng, "IceArch", 240,
                    (x, z) => SampleBiome(x, z) == Biome.Snow);

            // THE glacier: in granite country, a tongue of ice flows from
            // the snowfield edge down its valley.
            if (settings.Profile == LandformProfile.AlpineGranite)
            {
                TryPlaceWonder(rng, "GlacierFlow", 300, (x, z) =>
                {
                    if (SampleBiome(x, z) != Biome.Snow)
                        return false;
                    const float e = 6f;
                    float gx = SampleHeight(x + e, z) - SampleHeight(x - e, z);
                    float gz = SampleHeight(x, z + e) - SampleHeight(x, z - e);
                    float slope = Mathf.Sqrt(gx * gx + gz * gz) / (2f * e);
                    return slope > 0.14f && slope < 0.55f;   // a flowing valley side
                });
            }

            // THE crater: a lava lake in the caldera, and a flow down the
            // flank the placer walks with real downhill queries.
            if (_hasVolcano)
            {
                _wonders.Add(new WonderSite
                {
                    Kind = "LavaCrater",
                    Pos = VolcanoCrater,
                    YawDeg = (float)rng.NextDouble() * 360f,
                    Scale = 1f,
                });
                float flowAng = (float)rng.NextDouble() * Mathf.PI * 2f;
                var rim = _volcanoPos + new Vector2(Mathf.Cos(flowAng), Mathf.Sin(flowAng)) * 40f;
                _wonders.Add(new WonderSite
                {
                    Kind = "LavaFlow",
                    Pos = new Vector3(rim.x, SampleHeight(rim.x, rim.y), rim.y),
                    YawDeg = flowAng * Mathf.Rad2Deg,
                    Scale = 1f,
                });
            }

            // THE cave: bored into the tallest mossy cliff face — the shell
            // mesh provides the overhang a heightfield can't.
            int bestSeg = -1;
            float bestSegScale = 1.4f;
            if (!Roll(0.85f))
                bestSegScale = float.MaxValue;   // this island keeps no cave
            for (int i = 0; i < _inlandCliffs.Count; i++)
            {
                var c = _inlandCliffs[i];
                if (c.Family != CliffFamily.Mossy || c.Scale < bestSegScale)
                    continue;
                // The approach in front of the mouth must be dry, walkable land
                // (this rules out lake-bank cliffs whose feet stand in water).
                float crad = c.YawDeg * Mathf.Deg2Rad;
                var cdown = new Vector2(Mathf.Sin(crad), Mathf.Cos(crad));
                float ax = c.Pos.x + cdown.x * 7f, az = c.Pos.z + cdown.y * 7f;
                if (StandingWaterDepth(ax, az) > 0.05f
                    || SampleHeight(ax, az) < settings.SeaLevel + 1.5f
                    || RiverInfluence01(ax, az) > 0.2f)
                    continue;
                bool clear = true;
                foreach (var w in _wonders)
                {
                    float dx = c.Pos.x - w.Pos.x, dz = c.Pos.z - w.Pos.z;
                    if (dx * dx + dz * dz < 60f * 60f) { clear = false; break; }
                }
                if (clear) { bestSeg = i; bestSegScale = c.Scale; }
            }
            if (bestSeg >= 0)
            {
                var seg = _inlandCliffs[bestSeg];
                float rad = seg.YawDeg * Mathf.Deg2Rad;
                var down = new Vector2(Mathf.Sin(rad), Mathf.Cos(rad));
                var mouth = new Vector2(seg.Pos.x, seg.Pos.z);
                var center = mouth - down * 6f;     // pushed back into the hill
                float floorY = SampleHeight(mouth.x + down.x * 5f, mouth.y + down.y * 5f);
                CarveCaveFloor(center, 13f, floorY);
                _wonders.Add(new WonderSite
                {
                    Kind = "Cave",
                    Pos = new Vector3(center.x, floorY, center.y),
                    YawDeg = seg.YawDeg,
                    Scale = 2.2f,
                });
                // Clear crag pieces that would collide with the shell or
                // crowd the mouth — the same wall run continues on both sides.
                _inlandCliffs.RemoveAll(c =>
                    (c.Pos.x - center.x) * (c.Pos.x - center.x)
                    + (c.Pos.z - center.y) * (c.Pos.z - center.y) < 27f * 27f);
            }
        }

        /// <summary>
        /// Scoop the height grid down to the cave-floor level within the
        /// shell's footprint, so terrain never pokes up through the cavity.
        /// </summary>
        private void CarveCaveFloor(Vector2 center, float radius, float floorY)
        {
            float half = settings.WorldSize * 0.5f;
            int reach = Mathf.CeilToInt(radius / _heightCell) + 1;
            int cgx = Mathf.RoundToInt((center.x + half) / _heightCell);
            int cgz = Mathf.RoundToInt((center.y + half) / _heightCell);
            for (int dz = -reach; dz <= reach; dz++)
            {
                for (int dx = -reach; dx <= reach; dx++)
                {
                    int gx = cgx + dx, gz = cgz + dz;
                    if (gx < 0 || gz < 0 || gx >= _heightRes || gz >= _heightRes)
                        continue;
                    float d = Mathf.Sqrt(dx * dx + dz * dz) * _heightCell;
                    if (d > radius)
                        continue;
                    float t = Mathf.SmoothStep(1f, 0f, d / radius);
                    float target = Mathf.Lerp(_heightGrid[gx, gz], floorY, t);
                    if (target < _heightGrid[gx, gz])
                        _heightGrid[gx, gz] = target;
                }
            }
        }

        private bool TryPlaceWonder(System.Random rng, string kind, int attempts,
            System.Func<float, float, bool> suitable)
        {
            float half = settings.WorldSize * 0.5f;
            for (int i = 0; i < attempts; i++)
            {
                float x = ((float)rng.NextDouble() * 2f - 1f) * half * 0.92f;
                float z = ((float)rng.NextDouble() * 2f - 1f) * half * 0.92f;
                if (!suitable(x, z))
                    continue;
                _wonders.Add(new WonderSite
                {
                    Kind = kind,
                    Pos = new Vector3(x, SampleHeight(x, z), z),
                    YawDeg = (float)rng.NextDouble() * 360f,
                    Scale = 1f,
                });
                return true;
            }
            return false;
        }

        /// <summary>Domain warp (Quilez): organic boundaries from warped inputs.</summary>
        private Vector2 Warp(float x, float z)
        {
            float f = 1f / settings.WarpFeatureSize;
            float wx = (_warpNoiseX.Sample(x, z, f) - 0.5f) * 2f * settings.WarpStrength;
            float wz = (_warpNoiseZ.Sample(x, z, f) - 0.5f) * 2f * settings.WarpStrength;
            return new Vector2(x + wx, z + wz);
        }

        [ContextMenu("Generate")]
        public void Generate()
        {
            if (settings == null)
            {
                Debug.LogError("[AniML] ProceduralWorld has no WorldSettings.");
                return;
            }

            _heightNoise = null;
            EnsureSamplers();

            ClearChunks();

            int chunksPerSide = Mathf.RoundToInt(settings.WorldSize / settings.ChunkSize);
            float half = settings.WorldSize * 0.5f;

            for (int cz = 0; cz < chunksPerSide; cz++)
            {
                for (int cx = 0; cx < chunksPerSide; cx++)
                {
                    var origin = new Vector3(cx * settings.ChunkSize - half, 0f, cz * settings.ChunkSize - half);
                    BuildChunk(cx, cz, origin);
                }
            }

            BuildRiverWater();
        }

        // ---- World truth functions (sim + scatter + brushes read these) ----

        public float SampleHeight(float x, float z)
        {
            if (_gridReady)
            {
                float half = settings.WorldSize * 0.5f;
                float gx = Mathf.Clamp((x + half) / _heightCell, 0f, _heightRes - 1.001f);
                float gz = Mathf.Clamp((z + half) / _heightCell, 0f, _heightRes - 1.001f);
                int x0 = (int)gx, z0 = (int)gz;
                float fx = gx - x0, fz = gz - z0;
                return Mathf.Lerp(
                    Mathf.Lerp(_heightGrid[x0, z0], _heightGrid[x0 + 1, z0], fx),
                    Mathf.Lerp(_heightGrid[x0, z0 + 1], _heightGrid[x0 + 1, z0 + 1], fx), fz);
            }
            return AnalyticHeight(x, z);
        }

        private float AnalyticHeight(float x, float z)
        {
            float raw = RawHeight(x, z);
            GetRiverData(x, z, out float influence, out float bed);

            // Fords: near a ford stamp the carve target rises to a gravel
            // bar ~0.35 m under the water surface (surface = bed + 55% of
            // RiverDepth, and it never moves — so no uphill-water artifacts,
            // just a wadeable riffle).
            if (influence > 0f)
            {
                foreach (var ford in _fords)
                {
                    float fdx = x - ford.Pos.x, fdz = z - ford.Pos.y;
                    float fd = Mathf.Sqrt(fdx * fdx + fdz * fdz);
                    if (fd >= ford.Radius)
                        continue;
                    float lift = Mathf.SmoothStep(0f, 1f, 1f - fd / ford.Radius)
                                 * Mathf.Max(settings.RiverDepth * 0.55f - 0.35f, 0f);
                    bed += lift;
                }
            }

            // Deep carves (the canyon) get sharpened influence → near-vertical
            // walls instead of gentle banks.
            if (influence > 0f && raw - bed > 7f)
            {
                influence = Mathf.Pow(influence, 0.45f);

                // Truly deep walls terrace like layered sedimentary rock:
                // flat treads stepping down through smooth risers.
                if (raw - bed > 14f)
                {
                    const float steps = 4f;
                    float t = influence * steps;
                    float f = t - Mathf.Floor(t);
                    float stepped = (Mathf.Floor(t)
                        + Mathf.SmoothStep(0f, 1f, Mathf.Clamp01((f - 0.35f) / 0.3f))) / steps;
                    influence = Mathf.Lerp(influence, stepped, 0.8f);
                }
            }
            float h = influence <= 0f ? raw
                // Blend down toward the (monotonically descending) river bed;
                // min() so banks in local dips are never raised.
                : Mathf.Lerp(raw, Mathf.Min(bed, raw), influence);

            // Pond basins: shallow scooped depressions.
            foreach (var pond in _ponds)
            {
                float dx = x - pond.Pos.x, dz = z - pond.Pos.y;
                float d = Mathf.Sqrt(dx * dx + dz * dz);
                if (d < pond.Radius)
                {
                    float t = Mathf.SmoothStep(0f, 1f, 1f - d / pond.Radius);
                    h = Mathf.Lerp(h, Mathf.Min(pond.BottomY, h), t);
                }
            }

            // The canyon crossing: a talus ramp corridor cut down through
            // both rims to the center ford. Only ever lowers terrain, and
            // its base sits just above the channel so the ford's gravel
            // bar (not this ramp) sets the crossing depth.
            if (_gorgeCross.Active)
            {
                float cdx = x - _gorgeCross.Pos.x, cdz = z - _gorgeCross.Pos.y;
                float u = cdx * _gorgeCross.Dir.x + cdz * _gorgeCross.Dir.y;
                float v = cdx * _gorgeCross.RiverDir.x + cdz * _gorgeCross.RiverDir.y;
                float au = Mathf.Abs(u), av = Mathf.Abs(v);
                if (au < _gorgeCross.HalfLen && av < _gorgeCross.HalfWidth * 2.2f)
                {
                    float rim = u > 0f ? _gorgeCross.RimB : _gorgeCross.RimA;
                    float ramp = Mathf.Lerp(_gorgeCross.BedY + 1.35f, rim + 1.5f,
                        Mathf.SmoothStep(0f, 1f, au / _gorgeCross.HalfLen));
                    float edge = Mathf.SmoothStep(1f, 0f,
                        Mathf.InverseLerp(_gorgeCross.HalfWidth, _gorgeCross.HalfWidth * 2.2f, av));
                    if (ramp < h)
                        h = Mathf.Lerp(h, ramp, edge);
                }
            }
            return h;
        }

        /// <summary>
        /// Rasterize the analytic height (with river/pond carves) into a grid
        /// aligned with the terrain mesh quads, then run thermal erosion:
        /// slopes steeper than the talus angle shed material downhill,
        /// producing scree shoulders and settled, weathered ridges. River
        /// channels are masked so erosion never breaks the carved beds.
        /// </summary>
        private void BakeHeightGrid()
        {
            float half = settings.WorldSize * 0.5f;
            _heightCell = settings.QuadSize;
            _heightRes = Mathf.RoundToInt(settings.WorldSize / _heightCell) + 1;
            _heightGrid = new float[_heightRes, _heightRes];
            var riverMask = new bool[_heightRes, _heightRes];

            for (int gz = 0; gz < _heightRes; gz++)
            {
                for (int gx = 0; gx < _heightRes; gx++)
                {
                    float x = gx * _heightCell - half;
                    float z = gz * _heightCell - half;
                    _heightGrid[gx, gz] = AnalyticHeight(x, z);
                    riverMask[gx, gz] = RiverInfluence01(x, z) > 0.25f;
                }
            }

            // Hydraulic droplet erosion: rain carves the drainage thermal
            // relaxation can't — dendritic tributary creases feeding the
            // rivers, sharpened ridges, sediment fans where slopes meet the
            // plains. Runs BEFORE the landform stamps (pedestals, volcano,
            // cave floors), which overwrite it wherever drama is authored.
            // Hard rock (badlands/mesa country) erodes far less, so crisp
            // features stay crisp. Deterministic per seed.
            {
                var hardness = new float[_heightRes, _heightRes];
                for (int gz = 0; gz < _heightRes; gz++)
                {
                    for (int gx = 0; gx < _heightRes; gx++)
                    {
                        float x = gx * _heightCell - half;
                        float z = gz * _heightCell - half;
                        hardness[gx, gz] = Mathf.Clamp01(SampleRockiness01(x, z) * 1.25f);
                    }
                }

                var rng = new System.Random(settings.Seed * 197 + 3);
                _sedimentOutflow = new float[_heightRes, _heightRes];
                int droplets = (_heightRes * _heightRes) / 7;
                const float inertia = 0.06f;
                const float capacityK = 3.6f;
                const float minCapacity = 0.01f;
                const float erodeK = 0.28f;
                const float depositK = 0.28f;
                const float evaporate = 0.015f;
                const float gravity = 4f;
                const int maxSteps = 42;

                for (int d = 0; d < droplets; d++)
                {
                    float px = 1f + (float)rng.NextDouble() * (_heightRes - 3);
                    float pz = 1f + (float)rng.NextDouble() * (_heightRes - 3);
                    float dirX = 0f, dirZ = 0f, speed = 1f, water = 1f, sediment = 0f;

                    for (int step = 0; step < maxSteps; step++)
                    {
                        int ix = (int)px, iz = (int)pz;
                        float fx = px - ix, fz = pz - iz;

                        float h00 = _heightGrid[ix, iz], h10 = _heightGrid[ix + 1, iz];
                        float h01 = _heightGrid[ix, iz + 1], h11 = _heightGrid[ix + 1, iz + 1];
                        float gradX = (h10 - h00) * (1f - fz) + (h11 - h01) * fz;
                        float gradZ = (h01 - h00) * (1f - fx) + (h11 - h10) * fx;
                        float height = h00 * (1f - fx) * (1f - fz) + h10 * fx * (1f - fz)
                                       + h01 * (1f - fx) * fz + h11 * fx * fz;

                        dirX = dirX * inertia - gradX * (1f - inertia);
                        dirZ = dirZ * inertia - gradZ * (1f - inertia);
                        float len = Mathf.Sqrt(dirX * dirX + dirZ * dirZ);
                        if (len < 1e-5f)
                            break;
                        dirX /= len;
                        dirZ /= len;
                        px += dirX;
                        pz += dirZ;
                        if (px < 1f || pz < 1f || px >= _heightRes - 2 || pz >= _heightRes - 2)
                            break;

                        int nx = (int)px, nz = (int)pz;
                        float nfx = px - nx, nfz = pz - nz;
                        float nh = _heightGrid[nx, nz] * (1f - nfx) * (1f - nfz)
                                   + _heightGrid[nx + 1, nz] * nfx * (1f - nfz)
                                   + _heightGrid[nx, nz + 1] * (1f - nfx) * nfz
                                   + _heightGrid[nx + 1, nz + 1] * nfx * nfz;
                        float dh = nh - height;

                        // Droplets end at carved channels and the sea — and
                        // hand over whatever sediment they still carry, so
                        // the coast can build deltas from it (conservation).
                        if (riverMask[ix, iz] || nh <= settings.SeaLevel + 0.5f)
                        {
                            _sedimentOutflow[ix, iz] += sediment;
                            break;
                        }

                        float capacity = Mathf.Max(-dh * speed * water * capacityK, minCapacity);
                        if (sediment > capacity || dh > 0f)
                        {
                            float amount = dh > 0f
                                ? Mathf.Min(dh, sediment)
                                : (sediment - capacity) * depositK;
                            sediment -= amount;
                            _heightGrid[ix, iz] += amount * (1f - fx) * (1f - fz);
                            _heightGrid[ix + 1, iz] += amount * fx * (1f - fz);
                            _heightGrid[ix, iz + 1] += amount * (1f - fx) * fz;
                            _heightGrid[ix + 1, iz + 1] += amount * fx * fz;
                        }
                        else
                        {
                            float soft = 1f - hardness[ix, iz] * 0.82f;
                            float amount = Mathf.Min((capacity - sediment) * erodeK * soft, -dh);
                            sediment += amount;
                            _heightGrid[ix, iz] -= amount * (1f - fx) * (1f - fz);
                            _heightGrid[ix + 1, iz] -= amount * fx * (1f - fz);
                            _heightGrid[ix, iz + 1] -= amount * (1f - fx) * fz;
                            _heightGrid[ix + 1, iz + 1] -= amount * fx * fz;
                        }

                        speed = Mathf.Sqrt(Mathf.Max(speed * speed - dh * gravity, 0f));
                        water *= 1f - evaporate;
                        if (water < 0.05f)
                            break;
                    }
                }
            }

            // Sediment closure: everything the rain carried to the coast is
            // deposited as shallow fans — deltas and bars at the mouths,
            // visible through the depth-faded water. Smoothed twice so fans
            // spread naturally.
            for (int pass = 0; pass < 2; pass++)
            {
                var blurred = new float[_heightRes, _heightRes];
                for (int gz = 1; gz < _heightRes - 1; gz++)
                {
                    for (int gx = 1; gx < _heightRes - 1; gx++)
                    {
                        blurred[gx, gz] = (_sedimentOutflow[gx, gz]
                            + _sedimentOutflow[gx - 1, gz] + _sedimentOutflow[gx + 1, gz]
                            + _sedimentOutflow[gx, gz - 1] + _sedimentOutflow[gx, gz + 1]) / 5f;
                    }
                }
                _sedimentOutflow = blurred;
            }
            for (int gz = 0; gz < _heightRes; gz++)
            {
                for (int gx = 0; gx < _heightRes; gx++)
                {
                    if (_sedimentOutflow[gx, gz] <= 0.01f)
                        continue;
                    if (_heightGrid[gx, gz] < settings.SeaLevel + 0.2f)
                        _heightGrid[gx, gz] += Mathf.Min(_sedimentOutflow[gx, gz] * 0.8f, 1.7f);
                }
            }

            // Fill the single-cell pits droplets leave when they die mid-slope
            // (otherwise priority-flood turns each into a perched puddle).
            for (int pass = 0; pass < 2; pass++)
            {
                for (int gz = 1; gz < _heightRes - 1; gz++)
                {
                    for (int gx = 1; gx < _heightRes - 1; gx++)
                    {
                        float h = _heightGrid[gx, gz];
                        float minN = Mathf.Min(
                            Mathf.Min(_heightGrid[gx - 1, gz], _heightGrid[gx + 1, gz]),
                            Mathf.Min(_heightGrid[gx, gz - 1], _heightGrid[gx, gz + 1]));
                        if (h < minN - 0.04f)
                            _heightGrid[gx, gz] = minN - 0.02f;
                    }
                }
            }

            // Dune fields: on dry soft flats, sand piles into ridges running
            // across the wind, gentle on the windward side.
            if (_dials.Dunes)
            {
                var wind = settings.WindDirection.normalized;
                for (int gz = 0; gz < _heightRes; gz++)
                {
                    for (int gx = 0; gx < _heightRes; gx++)
                    {
                        float h = _heightGrid[gx, gz] - settings.SeaLevel;
                        if (h < 2f || h > 20f)
                            continue;
                        float x = gx * _heightCell - half;
                        float z = gz * _heightCell - half;
                        if (SampleRockiness01(x, z) > 0.22f)
                            continue;
                        float mask = Mathf.InverseLerp(0.46f, 0.62f,
                            _duneNoise.Sample(x, z, 1f / 230f));
                        if (mask <= 0f)
                            continue;
                        // Skewed wave: long windward back, steep lee face.
                        float phase = (x * wind.x + z * wind.y) * 0.14f
                                      + _duneNoise.Sample(z, x, 1f / 90f) * 4f;
                        float wave = Mathf.Pow(0.5f + 0.5f * Mathf.Sin(phase + 0.65f * Mathf.Sin(phase)), 1.5f);
                        _heightGrid[gx, gz] += wave * 2.2f * mask;
                    }
                }
            }

            // Glacial carve: above the paleo-snowline the old ice widened
            // and flattened the valleys (U-profiles) and left a moraine
            // ridge where it stopped. Ridges survive; floors broaden.
            if (_dials.GlacialCarve)
            {
                float paleo = settings.SeaLevel + _dials.PaleoSnowline;
                for (int pass = 0; pass < 6; pass++)
                {
                    for (int gz = 2; gz < _heightRes - 2; gz++)
                    {
                        for (int gx = 2; gx < _heightRes - 2; gx++)
                        {
                            float h = _heightGrid[gx, gz];
                            if (h < paleo)
                                continue;
                            float mean = 0f, min = float.MaxValue;
                            for (int dz = -2; dz <= 2; dz += 2)
                            {
                                for (int dx = -2; dx <= 2; dx += 2)
                                {
                                    float nh2 = _heightGrid[gx + dx, gz + dz];
                                    mean += nh2;
                                    if (nh2 < min) min = nh2;
                                }
                            }
                            mean /= 9f;
                            if (h <= mean)   // valley floor or lower wall
                                _heightGrid[gx, gz] =
                                    Mathf.Lerp(h, min + (h - min) * 0.35f, 0.4f);
                        }
                    }
                }
                // The terminal moraine: a low rubble ridge along the old
                // ice margin, on gentle ground only.
                for (int gz = 2; gz < _heightRes - 2; gz++)
                {
                    for (int gx = 2; gx < _heightRes - 2; gx++)
                    {
                        float h = _heightGrid[gx, gz];
                        if (Mathf.Abs(h - paleo) > 1.3f)
                            continue;
                        float slope = Mathf.Abs(_heightGrid[gx + 1, gz] - _heightGrid[gx - 1, gz])
                                      + Mathf.Abs(_heightGrid[gx, gz + 1] - _heightGrid[gx, gz - 1]);
                        if (slope < 1.4f)
                            _heightGrid[gx, gz] += 1.5f;
                    }
                }
            }

            // Tide shelf: compress the just-underwater band toward a shallow
            // flat, so gentle coasts get a wide wading littoral (seaweed,
            // coral, and tide rocks live here) instead of a steep ramp.
            for (int gz = 0; gz < _heightRes; gz++)
            {
                for (int gx = 0; gx < _heightRes; gx++)
                {
                    float h = _heightGrid[gx, gz];
                    float t = Mathf.InverseLerp(settings.SeaLevel - 2.4f, settings.SeaLevel, h);
                    if (t <= 0f || t >= 1f)
                        continue;
                    float target = Mathf.Lerp(settings.SeaLevel - 1.0f, settings.SeaLevel - 0.15f, t * t);
                    _heightGrid[gx, gz] = Mathf.Lerp(h, target, 0.6f);
                }
            }

            // Thermal erosion: talus angle ~33 degrees.
            float talus = Mathf.Tan(33f * Mathf.Deg2Rad) * _heightCell;
            const int iterations = 26;
            for (int it = 0; it < iterations; it++)
            {
                for (int gz = 1; gz < _heightRes - 1; gz++)
                {
                    for (int gx = 1; gx < _heightRes - 1; gx++)
                    {
                        if (riverMask[gx, gz]) continue;

                        float h = _heightGrid[gx, gz];
                        int lx = gx, lz = gz;
                        float lowest = h;
                        if (_heightGrid[gx - 1, gz] < lowest) { lowest = _heightGrid[gx - 1, gz]; lx = gx - 1; lz = gz; }
                        if (_heightGrid[gx + 1, gz] < lowest) { lowest = _heightGrid[gx + 1, gz]; lx = gx + 1; lz = gz; }
                        if (_heightGrid[gx, gz - 1] < lowest) { lowest = _heightGrid[gx, gz - 1]; lx = gx; lz = gz - 1; }
                        if (_heightGrid[gx, gz + 1] < lowest) { lowest = _heightGrid[gx, gz + 1]; lx = gx; lz = gz + 1; }

                        float drop = h - lowest;
                        if (drop > talus && !riverMask[lx, lz])
                        {
                            float move = 0.18f * (drop - talus);
                            _heightGrid[gx, gz] -= move;
                            _heightGrid[lx, lz] += move;
                        }
                    }
                }
            }

            // Karst: the soluble ground opens sinkholes — parabolic bowls on
            // gentle mid-elevation ground away from rivers. The broad, deep
            // ones flood via the ordinary lake detector into karst ponds.
            if (_dials.KarstCount > 0)
            {
                var karstRng = new System.Random(settings.Seed * 233 + 41);
                int placed = 0;
                for (int attempt = 0; attempt < 260 && placed < _dials.KarstCount; attempt++)
                {
                    float x = ((float)karstRng.NextDouble() * 2f - 1f) * half * 0.7f;
                    float z = ((float)karstRng.NextDouble() * 2f - 1f) * half * 0.7f;
                    int cgx = Mathf.RoundToInt((x + half) / _heightCell);
                    int cgz = Mathf.RoundToInt((z + half) / _heightCell);
                    if (cgx < 12 || cgz < 12 || cgx >= _heightRes - 12 || cgz >= _heightRes - 12)
                        continue;
                    float ch = _heightGrid[cgx, cgz] - settings.SeaLevel;
                    if (ch < 6f || ch > 30f)
                        continue;
                    if (RiverInfluence01(x, z) > 0f || SampleRockiness01(x, z) > 0.4f)
                        continue;
                    float spread = Mathf.Abs(_heightGrid[cgx + 5, cgz] - _heightGrid[cgx - 5, cgz])
                                   + Mathf.Abs(_heightGrid[cgx, cgz + 5] - _heightGrid[cgx, cgz - 5]);
                    if (spread > 3.5f)
                        continue;

                    float radius = 14f + (float)karstRng.NextDouble() * 5f;
                    float depth = 3.2f + (float)karstRng.NextDouble() * 1.2f;
                    float centerY = _heightGrid[cgx, cgz];
                    int reach = Mathf.CeilToInt(radius / _heightCell);
                    for (int dz = -reach; dz <= reach; dz++)
                    {
                        for (int dx = -reach; dx <= reach; dx++)
                        {
                            float d = Mathf.Sqrt(dx * dx + dz * dz) * _heightCell;
                            if (d > radius)
                                continue;
                            float t = d / radius;
                            float target = centerY - depth * Mathf.Pow(1f - t * t, 1.2f);
                            int gx2 = cgx + dx, gz2 = cgz + dz;
                            if (target < _heightGrid[gx2, gz2])
                                _heightGrid[gx2, gz2] = Mathf.Lerp(_heightGrid[gx2, gz2], target,
                                    Mathf.SmoothStep(1f, 0.75f, t));
                        }
                    }
                    placed++;
                }
            }

            _gridReady = true;
        }

        /// <summary>
        /// Priority-flood (Barnes et al.): flood the grid inward from the map
        /// border with a min-heap; any cell that must be raised to reach an
        /// outlet is part of a depression. Contiguous flooded cells become
        /// lakes with a shared water level — water exists exactly where the
        /// terrain would actually hold it.
        /// </summary>
        private void DetectLakes()
        {
            _lakes.Clear();
            _lakeCellIndex.Clear();

            int res = _heightRes;
            var fill = new float[res, res];
            var visited = new bool[res, res];
            var heap = new List<(float h, int gx, int gz)>();

            void HeapPush((float h, int gx, int gz) item)
            {
                heap.Add(item);
                int i = heap.Count - 1;
                while (i > 0)
                {
                    int parent = (i - 1) / 2;
                    if (heap[parent].h <= heap[i].h) break;
                    (heap[parent], heap[i]) = (heap[i], heap[parent]);
                    i = parent;
                }
            }

            (float h, int gx, int gz) HeapPop()
            {
                var top = heap[0];
                heap[0] = heap[^1];
                heap.RemoveAt(heap.Count - 1);
                int i = 0;
                while (true)
                {
                    int l = i * 2 + 1, r = l + 1, m = i;
                    if (l < heap.Count && heap[l].h < heap[m].h) m = l;
                    if (r < heap.Count && heap[r].h < heap[m].h) m = r;
                    if (m == i) break;
                    (heap[m], heap[i]) = (heap[i], heap[m]);
                    i = m;
                }
                return top;
            }

            for (int i = 0; i < res; i++)
            {
                foreach (var (gx, gz) in new[] { (i, 0), (i, res - 1), (0, i), (res - 1, i) })
                {
                    if (!visited[gx, gz])
                    {
                        visited[gx, gz] = true;
                        fill[gx, gz] = _heightGrid[gx, gz];
                        HeapPush((fill[gx, gz], gx, gz));
                    }
                }
            }

            var nbors = new (int dx, int dz)[] { (1, 0), (-1, 0), (0, 1), (0, -1) };
            while (heap.Count > 0)
            {
                var (h, gx, gz) = HeapPop();
                foreach (var (dx, dz) in nbors)
                {
                    int nx = gx + dx, nz = gz + dz;
                    if (nx < 0 || nz < 0 || nx >= res || nz >= res || visited[nx, nz])
                        continue;
                    visited[nx, nz] = true;
                    fill[nx, nz] = Mathf.Max(_heightGrid[nx, nz], h);
                    HeapPush((fill[nx, nz], nx, nz));
                }
            }

            // Group flooded cells (fill above ground and above sea) into lakes.
            var grouped = new bool[res, res];
            for (int gz = 1; gz < res - 1; gz++)
            {
                for (int gx = 1; gx < res - 1; gx++)
                {
                    if (grouped[gx, gz]) continue;
                    float depth = fill[gx, gz] - _heightGrid[gx, gz];
                    if (depth < 0.35f || fill[gx, gz] <= settings.SeaLevel + 0.3f) continue;

                    var lake = new Lake { WaterLevel = fill[gx, gz] };
                    var stack = new Stack<(int, int)>();
                    stack.Push((gx, gz));
                    grouped[gx, gz] = true;
                    float maxDepth = 0f;
                    while (stack.Count > 0)
                    {
                        var (cx, cz) = stack.Pop();
                        lake.Cells.Add((cx, cz));
                        maxDepth = Mathf.Max(maxDepth, fill[cx, cz] - _heightGrid[cx, cz]);
                        foreach (var (dx, dz) in new[] { (1, 0), (-1, 0), (0, 1), (0, -1) })
                        {
                            int nx = cx + dx, nz = cz + dz;
                            if (nx < 1 || nz < 1 || nx >= res - 1 || nz >= res - 1 || grouped[nx, nz])
                                continue;
                            if (fill[nx, nz] - _heightGrid[nx, nz] >= 0.35f
                                && Mathf.Abs(fill[nx, nz] - lake.WaterLevel) < 0.05f)
                            {
                                grouped[nx, nz] = true;
                                stack.Push((nx, nz));
                            }
                        }
                    }

                    // Keep real lakes; discard puddles.
                    // Real basins are broad relative to their depth; deep,
                    // narrow candidates are erosion pits — leave them dry.
                    if (lake.Cells.Count >= 60 && maxDepth >= _dials.MinLakeDepth
                        && lake.Cells.Count > maxDepth * 45f)
                    {
                        foreach (var (cx, cz) in lake.Cells)
                            _lakeCellIndex[((long)cx << 32) ^ (uint)cz] = _lakes.Count;
                        _lakes.Add(lake);
                    }
                }
            }
        }

        public bool InsideLake(float x, float z)
        {
            if (_lakeCellIndex.Count == 0) return false;
            float half = settings.WorldSize * 0.5f;
            int gx = Mathf.RoundToInt((x + half) / _heightCell);
            int gz = Mathf.RoundToInt((z + half) / _heightCell);
            return _lakeCellIndex.ContainsKey(((long)gx << 32) ^ (uint)gz);
        }

        /// <summary>
        /// Depth of standing water (lakes + ponds) above the terrain here;
        /// 0 = dry ground. Rivers are handled via RiverInfluence01 instead.
        /// </summary>
        public float StandingWaterDepth(float x, float z)
        {
            float depth = 0f;
            if (_lakeCellIndex.Count > 0)
            {
                float half = settings.WorldSize * 0.5f;
                int gx = Mathf.RoundToInt((x + half) / _heightCell);
                int gz = Mathf.RoundToInt((z + half) / _heightCell);
                if (_lakeCellIndex.TryGetValue(((long)gx << 32) ^ (uint)gz, out int li))
                    depth = Mathf.Max(depth, _lakes[li].WaterLevel - SampleHeight(x, z));
            }
            foreach (var pond in _ponds)
            {
                float dx = x - pond.Pos.x, dz = z - pond.Pos.y;
                if (dx * dx + dz * dz < pond.Radius * pond.Radius)
                    depth = Mathf.Max(depth, pond.WaterY - SampleHeight(x, z));
            }
            return Mathf.Max(0f, depth);
        }

        private void PlacePonds()
        {
            _ponds.Clear();
            var rng = new System.Random(settings.Seed * 41 + 13);
            float half = settings.WorldSize * 0.5f;
            int attempts = 0;

            while (_ponds.Count < _dials.PondCount && attempts++ < 260)
            {
                float x = ((float)rng.NextDouble() * 2f - 1f) * half * 0.72f;
                float z = ((float)rng.NextDouble() * 2f - 1f) * half * 0.72f;

                float h = RawHeight(x, z) - settings.SeaLevel;
                if (h < 2.5f || h > 22f) continue;
                if (RiverInfluence01(x, z) > 0f) continue;

                // Flat ground only: ponds don't sit on hillsides.
                const float e = 10f;
                float spread = Mathf.Max(
                    Mathf.Abs(RawHeight(x + e, z) - RawHeight(x - e, z)),
                    Mathf.Abs(RawHeight(x, z + e) - RawHeight(x, z - e)));
                if (spread > 2.2f) continue;

                float radius = 14f + (float)rng.NextDouble() * 18f;
                float ground = RawHeight(x, z);
                _ponds.Add(new PondStamp
                {
                    Pos = new Vector2(x, z),
                    Radius = radius,
                    BottomY = ground - 2.3f,
                    WaterY = ground - 0.7f,
                });
            }

            // Oxbow lakes: cutoff ponds on the OUTSIDE of sharp lowland
            // meander bends — the crescent scars a mature river leaves.
            int oxbows = 0;
            foreach (var path in _rivers)
            {
                for (int i = 10; i + 10 < path.Count && oxbows < 3; i += 6)
                {
                    var a = new Vector2(path[i - 8].x, path[i - 8].z);
                    var b = new Vector2(path[i].x, path[i].z);
                    var c = new Vector2(path[i + 8].x, path[i + 8].z);
                    if (RawHeight(b.x, b.y) - settings.SeaLevel > 14f)
                        continue;   // lowlands only
                    var d1 = (b - a).normalized;
                    var d2 = (c - b).normalized;
                    float turn = Vector2.SignedAngle(d1, d2);
                    if (Mathf.Abs(turn) < 42f)
                        continue;
                    var side = new Vector2(-d1.y, d1.x) * Mathf.Sign(-turn);
                    var pos = b + side * (path[i].w * 2.6f + 6f);
                    if (RiverInfluence01(pos.x, pos.y) > 0f)
                        continue;
                    float oxGround = RawHeight(pos.x, pos.y);
                    if (oxGround - settings.SeaLevel < 1.5f)
                        continue;
                    _ponds.Add(new PondStamp
                    {
                        Pos = pos,
                        Radius = 9f + path[i].w * 0.8f,
                        BottomY = oxGround - 2f,
                        WaterY = oxGround - 0.6f,
                    });
                    oxbows++;
                    i += 30;   // one oxbow per bend at most
                }
            }
        }

        /// <summary>Terrain height before river carving (rivers march on this).</summary>
        private float RawHeight(float x, float z)
        {
            // Domain-warped coordinates: ridgelines meander, the coastline
            // stops being a smooth disc.
            var w = Warp(x, z);

            // Rolling base terrain, exponent-shaped so lowlands stay walkable
            // and highlands accelerate upward.
            float n = _heightNoise.Sample(w.x, w.y, 1f / settings.HeightFeatureSize);
            float baseHeight = Mathf.Pow(n, settings.HeightExponent) * settings.HeightAmplitude
                               * _dials.HeightMul;

            // Ridged mountain layer, present only where the low-frequency
            // mountain mask says so — gives one coherent range, not spikes
            // everywhere.
            float ridgeSrc = _mountainNoise.Sample(w.x, w.y, 1f / settings.MountainFeatureSize);
            float ridge = 1f - Mathf.Abs(2f * ridgeSrc - 1f);      // fold: peaks at 0.5
            ridge = Mathf.Pow(ridge, 1.5f);                          // sharpen crests, keep broad shoulders
            float maskSrc = _mountainMaskNoise.Sample(w.x, w.y, 1f / settings.MountainMaskFeatureSize);
            float maskThreshold = settings.MountainMaskThreshold + _dials.MaskThresholdDelta;
            float mountainMask = Mathf.SmoothStep(0f, 1f,
                Mathf.InverseLerp(maskThreshold, maskThreshold + 0.18f, maskSrc));
            float mountains = ridge * mountainMask * settings.MountainAmplitude * _dials.MountainAmpMul;

            // Rocky badlands: jagged high-frequency relief inside a
            // low-frequency rockiness mask — breaks the everything-smooth
            // uniformity with genuinely rugged zones.
            float rockiness = RockinessFromMask(w.x, w.y);
            if (rockiness > 0f)
            {
                float roughSrc = _mountainNoise.Sample(w.x + 4096f, w.y - 2048f, 1f / settings.RockRoughFeatureSize);
                float rough = (1f - Mathf.Abs(2f * roughSrc - 1f)) * settings.RockRoughAmplitude;
                baseHeight += rough * rockiness;

                // Canyonlands: badlands zones quantize into stepped mesas.
                if (_dials.MesaTerraces && rockiness > 0.15f)
                {
                    float stepped = Mathf.Round(baseHeight / 8f) * 8f;
                    baseHeight = Mathf.Lerp(baseHeight, stepped, Mathf.Min(rockiness, 0.85f));
                }
            }

            // The fault: one side of the island sits higher — a clean
            // tectonic step whose trace wanders like a real fault line. The
            // cliff tracer dresses the scarp face on its own.
            if (_dials.FaultScarpH > 0.01f && _faultNoise != null)
            {
                float dLine = (x - _faultPoint.x) * _faultNormal.x + (z - _faultPoint.y) * _faultNormal.y;
                float along = (x - _faultPoint.x) * _faultDir.x + (z - _faultPoint.y) * _faultDir.y;
                dLine += (_faultNoise.Sample(along, 17f, 1f / 150f) - 0.5f) * 70f;
                baseHeight += _dials.FaultScarpH
                              * Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(-9f, 9f, dLine));
            }

            // Island falloff on warped distance — an irregular coastline.
            // With landmass variety on, "distance from center" becomes
            // distance from the nearest of several weighted landmass lobes:
            // blob, crescent bay, peninsula, or twin isles with a strait.
            float half = settings.WorldSize * 0.5f;
            float d;
            if (_lobes == null || _lobes.Length == 0)
            {
                d = Mathf.Sqrt(w.x * w.x + w.y * w.y) / half;
            }
            else
            {
                d = float.MaxValue;
                for (int i = 0; i < _lobes.Length; i++)
                {
                    float dx = w.x - _lobes[i].x, dz = w.y - _lobes[i].y;
                    float ld = Mathf.Sqrt(dx * dx + dz * dz) / (half * _lobes[i].z);
                    if (ld < d) d = ld;
                }
                // Crescent: a negative lobe pushes the sea INTO the land.
                if (_bayLobe.z > 0.01f)
                {
                    float bx = w.x - _bayLobe.x, bz = w.y - _bayLobe.y;
                    float bd = Mathf.Sqrt(bx * bx + bz * bz) / (half * _bayLobe.z);
                    d = Mathf.Max(d, 1.15f - bd * 0.85f);
                }
            }
            float islandMask = Mathf.Clamp01(1f - Mathf.Pow(Mathf.Max(d, 0f), settings.IslandFalloffPower));

            return (baseHeight + mountains) * islandMask - (1f - islandMask) * settings.EdgeSinkDepth;
        }

        // Landmass lobes: xy = center, z = radius scale. Empty = classic blob.
        private Vector3[] _lobes;
        private Vector3 _bayLobe;

        /// <summary>Roll this seed's landmass topology.</summary>
        private void ResolveLandmass()
        {
            _lobes = null;
            _bayLobe = Vector3.zero;
            if (!settings.LandmassVariety)
                return;

            var rng = new System.Random(settings.Seed * 613 + 29);
            float half = settings.WorldSize * 0.5f;
            double roll = rng.NextDouble();
            float Ang() => (float)rng.NextDouble() * Mathf.PI * 2f;

            if (roll < 0.4)
            {
                // Classic blob — still the most common island.
            }
            else if (roll < 0.6)
            {
                // Crescent: a deep bay bites into one side.
                float a = Ang();
                _lobes = new[] { new Vector3(0f, 0f, 1f) };
                _bayLobe = new Vector3(Mathf.Cos(a) * half * 0.55f,
                    Mathf.Sin(a) * half * 0.55f, 0.5f + (float)rng.NextDouble() * 0.2f);
            }
            else if (roll < 0.8)
            {
                // Peninsula: the main mass plus a long reaching arm.
                float a = Ang();
                _lobes = new[]
                {
                    new Vector3(-Mathf.Cos(a) * half * 0.18f, -Mathf.Sin(a) * half * 0.18f, 0.88f),
                    new Vector3(Mathf.Cos(a) * half * 0.52f, Mathf.Sin(a) * half * 0.52f, 0.42f),
                };
            }
            else
            {
                // Twin isles: two lands and the strait between them.
                float a = Ang();
                float sep = 0.42f + (float)rng.NextDouble() * 0.12f;
                _lobes = new[]
                {
                    new Vector3(Mathf.Cos(a) * half * sep, Mathf.Sin(a) * half * sep,
                        0.62f + (float)rng.NextDouble() * 0.12f),
                    new Vector3(-Mathf.Cos(a) * half * sep, -Mathf.Sin(a) * half * sep,
                        0.5f + (float)rng.NextDouble() * 0.12f),
                };
            }
        }

        private float RockinessFromMask(float wx, float wz)
        {
            float src = _mountainMaskNoise.Sample(wx - 8192f, wz + 4096f, 1f / settings.RockinessFeatureSize);
            return Mathf.SmoothStep(0f, 1f,
                Mathf.InverseLerp(settings.RockinessThreshold, settings.RockinessThreshold + 0.16f, src));
        }

        /// <summary>0 = soft ground, 1 = badlands. Drives roughness, rock splat, rock scatter.</summary>
        public float SampleRockiness01(float x, float z)
        {
            var w = Warp(x, z);
            return RockinessFromMask(w.x, w.y);
        }

        // ---- Climate fields ----

        /// <summary>0 = coldest (north peaks), 1 = warmest (south sea level).</summary>
        public float SampleTemperature01(float x, float z)
        {
            float latitude = z / settings.WorldSize; // -0.5 south .. +0.5 north
            float h = Mathf.Max(0f, SampleHeight(x, z) - settings.SeaLevel);
            float t = 0.62f
                + _dials.TempOffset
                - latitude * settings.LatitudeTempRange
                - h * settings.AltitudeLapse
                + (_moistureNoise.Sample(x + 977f, z + 331f, 1f / 400f) - 0.5f) * 0.08f;
            return Mathf.Clamp01(t);
        }

        /// <summary>
        /// Bake moisture as a decaying distance-from-water field (sea +
        /// rivers) with an orographic rain-shadow subtraction. Coarse grid,
        /// sampled bilinearly; noise detail is added at query time.
        /// </summary>
        private void BakeMoistureGrid()
        {
            float half = settings.WorldSize * 0.5f;
            _moistureRes = Mathf.CeilToInt(settings.WorldSize / MoistureCellSize) + 1;
            _moistureGrid = new float[_moistureRes, _moistureRes];
            var dist = new float[_moistureRes, _moistureRes];

            var oceanDist = new float[_moistureRes, _moistureRes];

            // Seed distances: 0 at water cells, +inf elsewhere. Ocean gets its
            // own field so beaches only form on actual coastline.
            for (int gz = 0; gz < _moistureRes; gz++)
            {
                for (int gx = 0; gx < _moistureRes; gx++)
                {
                    float x = gx * MoistureCellSize - half;
                    float z = gz * MoistureCellSize - half;
                    bool ocean = RawHeight(x, z) <= settings.SeaLevel + 0.2f;
                    // FRESH water only — marshes form around rivers and lakes,
                    // never salt coastline.
                    bool fresh = RiverInfluence01(x, z) > 0.35f || InsideLake(x, z);
                    dist[gx, gz] = fresh ? 0f : float.MaxValue;
                    oceanDist[gx, gz] = ocean ? 0f : float.MaxValue;
                }
            }
            ChamferTransform(oceanDist);
            _oceanDistGrid = oceanDist;

            ChamferTransform(dist);
            _freshDistGrid = (float[,])dist.Clone();

            // Moisture = exp decay of water distance, minus rain shadow.
            var wind = settings.WindDirection.normalized;
            for (int gz = 0; gz < _moistureRes; gz++)
            {
                for (int gx = 0; gx < _moistureRes; gx++)
                {
                    float x = gx * MoistureCellSize - half;
                    float z = gz * MoistureCellSize - half;
                    // Fresh water dominates moisture; the sea contributes a
                    // narrow humid coastal band only.
                    float m = _dials.BaseMoisture
                              + _dials.FreshGain * Mathf.Exp(-dist[gx, gz] / _dials.MoistureDecay)
                              + 0.20f * Mathf.Exp(-oceanDist[gx, gz] / 90f);
                    m = Mathf.Min(m, 1f);

                    // Rain shadow: tall terrain UPWIND wrings out the rain.
                    float hHere = RawHeight(x, z);
                    float maxUpwind = 0f;
                    for (int s = 1; s <= 6; s++)
                    {
                        float ux = x - wind.x * s * 35f;
                        float uz = z - wind.y * s * 35f;
                        maxUpwind = Mathf.Max(maxUpwind, RawHeight(ux, uz) - hHere);
                    }
                    float shadow = Mathf.Clamp01((maxUpwind - 18f) / 45f) * _dials.RainShadow;
                    _moistureGrid[gx, gz] = Mathf.Clamp01(m - shadow);
                }
            }
        }

        /// <summary>Two-pass chamfer distance transform (approximate Euclidean).</summary>
        private void ChamferTransform(float[,] grid)
        {
            int res = _moistureRes;
            float d1 = MoistureCellSize, d2 = MoistureCellSize * 1.4142f;
            for (int gz = 0; gz < res; gz++)
                for (int gx = 0; gx < res; gx++)
                {
                    float d = grid[gx, gz];
                    if (gx > 0) d = Mathf.Min(d, grid[gx - 1, gz] + d1);
                    if (gz > 0) d = Mathf.Min(d, grid[gx, gz - 1] + d1);
                    if (gx > 0 && gz > 0) d = Mathf.Min(d, grid[gx - 1, gz - 1] + d2);
                    if (gx < res - 1 && gz > 0) d = Mathf.Min(d, grid[gx + 1, gz - 1] + d2);
                    grid[gx, gz] = d;
                }
            for (int gz = res - 1; gz >= 0; gz--)
                for (int gx = res - 1; gx >= 0; gx--)
                {
                    float d = grid[gx, gz];
                    if (gx < res - 1) d = Mathf.Min(d, grid[gx + 1, gz] + d1);
                    if (gz < res - 1) d = Mathf.Min(d, grid[gx, gz + 1] + d1);
                    if (gx < res - 1 && gz < res - 1) d = Mathf.Min(d, grid[gx + 1, gz + 1] + d2);
                    if (gx > 0 && gz < res - 1) d = Mathf.Min(d, grid[gx - 1, gz + 1] + d2);
                    grid[gx, gz] = d;
                }
        }

        private bool InsidePondWater(float x, float z)
        {
            foreach (var pond in _ponds)
            {
                float dx = x - pond.Pos.x, dz = z - pond.Pos.y;
                if (dx * dx + dz * dz < pond.Radius * pond.Radius * 0.64f)
                    return true;
            }
            return false;
        }

        /// <summary>Meters to the nearest fresh water (rivers, ponds).</summary>
        public float SampleFreshwaterDistance(float x, float z)
        {
            if (_freshDistGrid == null)
                return 9999f;
            float half = settings.WorldSize * 0.5f;
            float gx = Mathf.Clamp((x + half) / MoistureCellSize, 0f, _moistureRes - 1.001f);
            float gz = Mathf.Clamp((z + half) / MoistureCellSize, 0f, _moistureRes - 1.001f);
            int x0 = (int)gx, z0 = (int)gz;
            float fx = gx - x0, fz = gz - z0;
            return Mathf.Lerp(
                Mathf.Lerp(_freshDistGrid[x0, z0], _freshDistGrid[x0 + 1, z0], fx),
                Mathf.Lerp(_freshDistGrid[x0, z0 + 1], _freshDistGrid[x0 + 1, z0 + 1], fx), fz);
        }

        /// <summary>Meters to the nearest actual coastline (not rivers).</summary>
        public float SampleOceanDistance(float x, float z)
        {
            if (_oceanDistGrid == null)
                return 9999f;
            float half = settings.WorldSize * 0.5f;
            float gx = Mathf.Clamp((x + half) / MoistureCellSize, 0f, _moistureRes - 1.001f);
            float gz = Mathf.Clamp((z + half) / MoistureCellSize, 0f, _moistureRes - 1.001f);
            int x0 = (int)gx, z0 = (int)gz;
            float fx = gx - x0, fz = gz - z0;
            return Mathf.Lerp(
                Mathf.Lerp(_oceanDistGrid[x0, z0], _oceanDistGrid[x0 + 1, z0], fx),
                Mathf.Lerp(_oceanDistGrid[x0, z0 + 1], _oceanDistGrid[x0 + 1, z0 + 1], fx), fz);
        }

        public float SampleMoisture(float x, float z)
        {
            if (_moistureGrid == null)
                return 0.5f;

            // Bilinear sample of the baked water-distance field.
            float half = settings.WorldSize * 0.5f;
            float gx = Mathf.Clamp((x + half) / MoistureCellSize, 0f, _moistureRes - 1.001f);
            float gz = Mathf.Clamp((z + half) / MoistureCellSize, 0f, _moistureRes - 1.001f);
            int x0 = (int)gx, z0 = (int)gz;
            float fx = gx - x0, fz = gz - z0;
            float m =
                Mathf.Lerp(
                    Mathf.Lerp(_moistureGrid[x0, z0], _moistureGrid[x0 + 1, z0], fx),
                    Mathf.Lerp(_moistureGrid[x0, z0 + 1], _moistureGrid[x0 + 1, z0 + 1], fx), fz);

            // Warped noise detail so the field isn't glassy-smooth.
            var w = Warp(x, z);
            float detail = (_moistureNoise.Sample(w.x, w.y, 1f / settings.MoistureFeatureSize) - 0.5f)
                           * settings.MoistureNoiseAmount;
            return Mathf.Clamp01(m + detail);
        }

        // ---- Forests as objects ----

        private void PlaceForests()
        {
            _forestSeeds.Clear();
            _forestClearings.Clear();
            var rng = new System.Random(settings.Seed * 17 + 91);
            float half = settings.WorldSize * 0.5f;
            int attempts = 0;

            while (_forestSeeds.Count < _dials.ForestCount && attempts++ < 400)
            {
                float x = ((float)rng.NextDouble() * 2f - 1f) * half * 0.85f;
                float z = ((float)rng.NextDouble() * 2f - 1f) * half * 0.85f;

                float h = SampleHeight(x, z) - settings.SeaLevel;
                if (h < 4f || h > 34f) continue;
                if (SampleMoisture(x, z) < 0.42f) continue;   // forests grow where it's wet

                float radius = Mathf.Lerp(settings.ForestRadiusRange.x, settings.ForestRadiusRange.y,
                    (float)rng.NextDouble());
                _forestSeeds.Add(new ForestSeed
                {
                    Pos = new Vector2(x, z),
                    Radius = radius,
                    Strength = 0.85f + (float)rng.NextDouble() * 0.35f,
                });

                // Punch 1-2 clearings into bigger forests — glades, and later
                // the sim's grazing hotspots.
                int clearings = radius > 110f ? 2 : 1;
                for (int c = 0; c < clearings; c++)
                {
                    float angle = (float)rng.NextDouble() * Mathf.PI * 2f;
                    float r = radius * (0.25f + (float)rng.NextDouble() * 0.45f);
                    _forestClearings.Add(new ForestSeed
                    {
                        Pos = new Vector2(x + Mathf.Cos(angle) * r, z + Mathf.Sin(angle) * r),
                        Radius = 16f + (float)rng.NextDouble() * 26f,
                        Strength = 1f,
                    });
                }
            }
        }

        /// <summary>0 = open land, 1 = deep forest core. Drives tree scatter.</summary>
        public float ForestDensity01(float x, float z)
        {
            if (_forestSeeds.Count == 0)
                return 0f;

            // Warped position gives forests organic, lobed edges.
            var w = Warp(x, z);

            float density = 0f;
            foreach (var seed in _forestSeeds)
            {
                float dx = w.x - seed.Pos.x, dz = w.y - seed.Pos.y;
                float d = Mathf.Sqrt(dx * dx + dz * dz);
                if (d < seed.Radius)
                {
                    float t = 1f - d / seed.Radius;
                    density += Mathf.SmoothStep(0f, 1f, t) * seed.Strength;
                }
            }

            foreach (var clearing in _forestClearings)
            {
                float dx = w.x - clearing.Pos.x, dz = w.y - clearing.Pos.y;
                float d = Mathf.Sqrt(dx * dx + dz * dz);
                if (d < clearing.Radius)
                    density -= (1f - d / clearing.Radius) * 1.4f;
            }

            density = Mathf.Clamp01(density);

            // An old burn: the fire took the canopy; snags and char remain.
            density *= 1f - BurnScar01(x, z) * 0.96f;

            // Ecotone tail: below the cutoff, a sparse "parkland" fringe of
            // lone trees rings each forest instead of a hard edge — the
            // in-between the hard cutoff was deleting. Far from any forest,
            // density is already 0 and plains stay truly open.
            if (density < settings.ForestDensityCutoff)
                return density * (density / settings.ForestDensityCutoff) * 0.18f;
            return density;
        }

        // ---- World memory: the burn scar (a REAL fire, run at genesis) ----

        private Vector2 _burnScarPos;
        private float _burnScarRadius;
        private bool[,] _scarMask;
        private int _scarRes;
        private float _scarCell;
        public Vector3 BurnScarCenter =>
            new(_burnScarPos.x, SampleHeight(_burnScarPos.x, _burnScarPos.y), _burnScarPos.y);
        public float BurnScarRadius => _burnScarRadius;

        /// <summary>0 outside, 1 inside the old fire's actual burned area
        /// (smoothed over the sim cells).</summary>
        public float BurnScar01(float x, float z)
        {
            if (_scarMask == null)
                return 0f;
            float half = settings.WorldSize * 0.5f;
            int gx = Mathf.RoundToInt((x + half) / _scarCell);
            int gz = Mathf.RoundToInt((z + half) / _scarCell);
            int hits = 0;
            for (int dz = -1; dz <= 1; dz++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = gx + dx, nz = gz + dz;
                    if (nx >= 0 && nz >= 0 && nx < _scarRes && nz < _scarRes && _scarMask[nx, nz])
                        hits++;
                }
            }
            return hits / 9f;
        }

        /// <summary>
        /// The old burn is not painted on — worldgen IGNITES a real fire at
        /// the driest forest edge and runs the same FireRules the live
        /// wildfire system uses. Whatever actually burned is the scar.
        /// </summary>
        private void ComputeBurnScar()
        {
            _scarMask = null;
            _burnScarRadius = 0f;
            if (_forestSeeds.Count == 0)
                return;
            var rng = new System.Random(settings.Seed * 131 + 17);

            // Ignition: the driest, fueled forest-edge candidate.
            float bestScore = 0.3f;
            Vector2 ignition = default;
            bool found = false;
            foreach (var seed in _forestSeeds)
            {
                for (int i = 0; i < 6; i++)
                {
                    float ang = (float)rng.NextDouble() * Mathf.PI * 2f;
                    var p = seed.Pos + new Vector2(Mathf.Cos(ang), Mathf.Sin(ang)) * seed.Radius * 0.75f;
                    if (SampleHeight(p.x, p.y) - settings.SeaLevel < 3f)
                        continue;
                    float score = FireRules.Fuel(this, p.x, p.y)
                                  * FireRules.Dryness(this, p.x, p.y, 0f);
                    if (score > bestScore)
                    {
                        bestScore = score;
                        ignition = p;
                        found = true;
                    }
                }
            }
            if (!found)
                return;

            // Run the fire on a coarse grid until it dies or is capped.
            _scarRes = 128;
            _scarCell = settings.WorldSize / _scarRes;
            _scarMask = new bool[_scarRes, _scarRes];
            var state = new byte[_scarRes, _scarRes];   // 0 clear 1 burning 2 burned
            float half = settings.WorldSize * 0.5f;
            var wind = settings.WindDirection.normalized;
            var burning = new List<(int, int)>();
            int igx = Mathf.Clamp(Mathf.RoundToInt((ignition.x + half) / _scarCell), 1, _scarRes - 2);
            int igz = Mathf.Clamp(Mathf.RoundToInt((ignition.y + half) / _scarCell), 1, _scarRes - 2);
            state[igx, igz] = 1;
            burning.Add((igx, igz));
            int burnedTotal = 0;

            var nbors = new (int dx, int dz)[] { (1, 0), (-1, 0), (0, 1), (0, -1) };
            for (int tick = 0; tick < 50 && burning.Count > 0 && burnedTotal < 520; tick++)
            {
                var next = new List<(int, int)>();
                foreach (var (bx, bz) in burning)
                {
                    foreach (var (dx, dz) in nbors)
                    {
                        int nx = bx + dx, nz = bz + dz;
                        if (nx < 1 || nz < 1 || nx >= _scarRes - 1 || nz >= _scarRes - 1
                            || state[nx, nz] != 0)
                            continue;
                        float wx = nx * _scarCell - half;
                        float wz = nz * _scarCell - half;
                        float chance = FireRules.SpreadChance(
                            FireRules.Fuel(this, wx, wz),
                            FireRules.Dryness(this, wx, wz, 0f) * 0.85f,
                            dx * wind.x + dz * wind.y);
                        if (rng.NextDouble() < chance)
                        {
                            state[nx, nz] = 1;
                            next.Add((nx, nz));
                        }
                    }
                    state[bx, bz] = 2;
                    _scarMask[bx, bz] = true;
                    burnedTotal++;
                }
                burning = next;
            }

            _burnScarPos = ignition;
            _burnScarRadius = Mathf.Sqrt(burnedTotal * _scarCell * _scarCell / Mathf.PI);
        }

        // ---- Biome classification ----

        public Biome SampleBiome(float x, float z)
        {
            float height = SampleHeight(x, z);
            float h = height - settings.SeaLevel;

            if (RiverInfluence01(x, z) > 0.45f && h > -0.5f)
                return Biome.RiverChannel;
            if (h < 0f)
                return Biome.Ocean;

            float temp = SampleTemperature01(x, z);
            float moisture = SampleMoisture(x, z);

            // Beach only along the actual coastline — low inland basins are
            // wetland or meadow, not mysterious sand lakes.
            if (h < settings.SandBandTop)
            {
                if (SampleOceanDistance(x, z) < 30f)
                    return Biome.Beach;
                return SampleFreshwaterDistance(x, z) < _dials.WetlandReach ? Biome.Wetland : Biome.Meadow;
            }

            // Cold-shifted snowline: northern mountains hold snow lower.
            float snowLine = settings.RockBandTop + (temp - 0.4f) * 34f;
            if (h > snowLine)
                return Biome.Snow;
            if (h > settings.RockBandTop * 0.92f)
                return Biome.Rock;

            // Marshes are freshwater-fed lowlands, never salt coastline.
            if (SampleFreshwaterDistance(x, z) < _dials.WetlandReach && h < 12f)
                return Biome.Wetland;
            if (ForestDensity01(x, z) >= settings.ForestDensityCutoff && h < 36f)
                return Biome.ForestFloor;
            if (temp < 0.3f && h > settings.GrassBandTop * 0.55f)
                return Biome.AlpineMeadow;
            // Scrub is the genuinely dry (rain-shadow / far-from-water) tail,
            // and only where it's warm — cold-dry reads alpine instead.
            if (moisture < 0.3f)
                return temp > 0.42f ? Biome.Scrub : Biome.AlpineMeadow;
            return Biome.Meadow;
        }

        // ---- Rock outcrops (crag objects) ----

        private void PlaceOutcrops()
        {
            _outcropSeeds.Clear();
            var rng = new System.Random(settings.Seed * 23 + 7);
            float half = settings.WorldSize * 0.5f;
            int attempts = 0;

            while (_outcropSeeds.Count < 14 && attempts++ < 300)
            {
                float x = ((float)rng.NextDouble() * 2f - 1f) * half * 0.85f;
                float z = ((float)rng.NextDouble() * 2f - 1f) * half * 0.85f;
                float h = SampleHeight(x, z) - settings.SeaLevel;
                if (h < 5f || h > 70f) continue;
                // Not near the coast — shoreline rock is the coastal cliff
                // system's job, not random offshore boulders.
                if (SampleOceanDistance(x, z) < 70f) continue;
                // Crags favor already-rocky or steep ground but a few stand alone.
                bool rocky = SampleRockiness01(x, z) > 0.35f;
                if (!rocky && rng.NextDouble() > 0.25) continue;

                _outcropSeeds.Add(new ForestSeed
                {
                    Pos = new Vector2(x, z),
                    Radius = 18f + (float)rng.NextDouble() * 30f,
                    Strength = 0.9f + (float)rng.NextDouble() * 0.3f,
                });
            }
        }

        /// <summary>
        /// Walk the coastline by radial rays; where the land drops steeply
        /// into the sea for a sustained run, chain aligned cliff meshes into
        /// a wall (instead of random offshore outcrops).
        /// </summary>
        private void TraceCoastCliffs()
        {
            _coastCliffs.Clear();
            var rng = new System.Random(settings.Seed * 53 + 3);
            float half = settings.WorldSize * 0.5f;

            const int steps = 720;
            var steep = new bool[steps];
            var coastR = new float[steps];

            for (int i = 0; i < steps; i++)
            {
                float ang = i / (float)steps * Mathf.PI * 2f;
                float cx = Mathf.Cos(ang), cz = Mathf.Sin(ang);

                // March outward to find the coastline crossing.
                float r = half * 0.35f;
                float lastAbove = -1f;
                while (r < half)
                {
                    if (SampleHeight(cx * r, cz * r) > settings.SeaLevel) lastAbove = r;
                    else if (lastAbove > 0f) break;
                    r += 6f;
                }
                coastR[i] = lastAbove;
                if (lastAbove < 0f) continue;

                // Steep coast: still tall a short distance inland.
                float inland = SampleHeight(cx * (lastAbove - 11f), cz * (lastAbove - 11f)) - settings.SeaLevel;
                steep[i] = inland > 6.5f;
            }

            // Runs of >= 7 consecutive steep rays become cliff walls.
            int runStart = -1;
            for (int i = 0; i <= steps; i++)
            {
                bool s = i < steps && steep[i] && coastR[i] > 0f;
                if (s && runStart < 0) runStart = i;
                if (!s && runStart >= 0)
                {
                    if (i - runStart >= 7)
                    {
                        for (int j = runStart; j < i; j += 2)
                        {
                            float ang = j / (float)steps * Mathf.PI * 2f;
                            float r = coastR[j] - 3f;
                            var pos = new Vector3(Mathf.Cos(ang) * r,
                                settings.SeaLevel - 0.5f, Mathf.Sin(ang) * r);
                            // Face the sea: yaw follows the coast tangent.
                            float yaw = -ang * Mathf.Rad2Deg + 90f + ((float)rng.NextDouble() - 0.5f) * 14f;
                            float inland = SampleHeight(Mathf.Cos(ang) * (r - 8f), Mathf.Sin(ang) * (r - 8f))
                                           - settings.SeaLevel;
                            _coastCliffs.Add(new CoastCliff
                            {
                                Pos = pos,
                                YawDeg = yaw,
                                Scale = Mathf.Clamp(inland / 5.5f, 1.2f, 2.6f),
                            });
                        }

                        // Sea stacks: long steep runs shed a spire or two
                        // into the surf — the eroded remnants of the old
                        // cliff line.
                        if (i - runStart >= 12 && rng.NextDouble() < 0.65)
                        {
                            int j = (runStart + i) / 2;
                            float angS = j / (float)steps * Mathf.PI * 2f;
                            float rS = coastR[j] + 12f + (float)rng.NextDouble() * 16f;
                            _coastCliffs.Add(new CoastCliff
                            {
                                Pos = new Vector3(Mathf.Cos(angS) * rS,
                                    settings.SeaLevel - 3.5f, Mathf.Sin(angS) * rS),
                                YawDeg = (float)rng.NextDouble() * 360f,
                                Scale = 2.4f + (float)rng.NextDouble() * 1.2f,
                            });
                        }
                    }
                    runStart = -1;
                }
            }
        }

        /// <summary>
        /// Dress steep inland terrain with chained cliff walls. Seeds where
        /// the baked slope exceeds a threshold, then marches the contour
        /// (perpendicular to the gradient) in both directions, emitting a
        /// piece every few meters while the ground stays steep — so cliffs
        /// come out as continuous aligned walls, not sprinkled rocks. Rock
        /// family is picked once per wall from the climate fields.
        /// </summary>
        private void TraceInlandCliffs()
        {
            _inlandCliffs.Clear();
            var rng = new System.Random(settings.Seed * 59 + 11);
            float half = settings.WorldSize * 0.5f;
            float slopeOn = _dials.CliffSlopeOn;   // slope (deg) that seeds a wall
            float slopeOff = slopeOn - 8f;         // hysteresis: runs continue down to this
            const float step = 6.5f;               // meters between chained pieces

            var visited = new bool[_heightRes, _heightRes];

            float SlopeDegAt(float x, float z)
            {
                const float e = 2f;
                float dx = SampleHeight(x + e, z) - SampleHeight(x - e, z);
                float dz = SampleHeight(x, z + e) - SampleHeight(x, z - e);
                return Mathf.Atan(Mathf.Sqrt(dx * dx + dz * dz) / (2f * e)) * Mathf.Rad2Deg;
            }

            Vector2 Grad(float x, float z)
            {
                const float e = 2f;
                return new Vector2(
                    SampleHeight(x + e, z) - SampleHeight(x - e, z),
                    SampleHeight(x, z + e) - SampleHeight(x, z - e));
            }

            void Mark(float x, float z)
            {
                int cgx = Mathf.RoundToInt((x + half) / _heightCell);
                int cgz = Mathf.RoundToInt((z + half) / _heightCell);
                const int rad = 4;
                for (int dz = -rad; dz <= rad; dz++)
                {
                    for (int dx = -rad; dx <= rad; dx++)
                    {
                        int nx = cgx + dx, nz = cgz + dz;
                        if (nx >= 0 && nz >= 0 && nx < _heightRes && nz < _heightRes)
                            visited[nx, nz] = true;
                    }
                }
            }

            void Emit(float x, float z, CliffFamily fam)
            {
                var grad = Grad(x, z);
                if (grad.sqrMagnitude < 1e-6f)
                    return;
                var down = -grad.normalized;

                // Local relief across the wall sets the piece height.
                float hUp = SampleHeight(x - down.x * 9f, z - down.y * 9f);
                float hDown = SampleHeight(x + down.x * 9f, z + down.y * 9f);
                float drop = hUp - hDown;
                if (drop < 5f)
                    return;

                // Base the piece at the wall's foot, facing downhill.
                float footX = x + down.x * 4f, footZ = z + down.y * 4f;
                if (RiverInfluence01(footX, footZ) > 0.9f)
                    return;
                // Never wall off the canyon crossing corridor.
                if (InCanyonCrossing(footX, footZ, 5f))
                    return;
                float footY = SampleHeight(footX, footZ);
                if (footY < settings.SeaLevel + 0.5f)
                    return;

                // Sink proportional to wall size — the pack meshes have
                // mixed pivots, and on steep ground a shallow embed leaves
                // downhill edges floating in air.
                _inlandCliffs.Add(new InlandCliff
                {
                    Pos = new Vector3(footX, footY - (1.5f + drop * 0.17f), footZ),
                    YawDeg = Mathf.Atan2(down.x, down.y) * Mathf.Rad2Deg
                             + ((float)rng.NextDouble() - 0.5f) * 12f,
                    Scale = Mathf.Clamp(drop / 9f, 0.9f, 2.6f),
                    Family = fam,
                });

                // Stamp the whole fall-line column so parallel contour rows
                // of the same face don't stack into a rubble pile — one wall
                // band per face (very tall faces may terrace at ~40m).
                Mark(x, z);
                for (int s = 1; s <= 10; s++)
                {
                    float d = s * 4f;
                    Mark(x - down.x * d, z - down.y * d);
                    Mark(x + down.x * d, z + down.y * d);
                }
            }

            for (int gz = 3; gz < _heightRes - 3; gz += 2)
            {
                for (int gx = 3; gx < _heightRes - 3; gx += 2)
                {
                    if (visited[gx, gz])
                        continue;
                    float x = gx * _heightCell - half;
                    float z = gz * _heightCell - half;
                    if (SlopeDegAt(x, z) < slopeOn)
                        continue;
                    float h = SampleHeight(x, z);
                    if (h < settings.SeaLevel + 2f)
                        continue;
                    if (SampleOceanDistance(x, z) < 45f)  // the coast tracer's turf
                        continue;
                    if (StandingWaterDepth(x, z) > 0.05f)
                        continue;

                    // One family per wall keeps a run coherent.
                    var biome = SampleBiome(x, z);
                    CliffFamily fam;
                    if (biome == Biome.Snow || biome == Biome.Rock || biome == Biome.AlpineMeadow
                        || h - settings.SeaLevel > _dials.GraniteAltitude)
                        fam = CliffFamily.Granite;
                    else if (biome == Biome.Scrub
                             || (SampleMoisture(x, z) < 0.36f && SampleTemperature01(x, z) > 0.5f))
                        fam = CliffFamily.RedRock;
                    else
                        fam = CliffFamily.Mossy;

                    // Canyon walls are always red rock, whatever grows above.
                    GetRiverData(x, z, out float rinf, out float rbed);
                    if (rinf > 0.15f && h - rbed > 10f)
                        fam = CliffFamily.RedRock;

                    // Collect the run: seed + contour march both directions.
                    var run = new List<Vector2> { new(x, z) };
                    foreach (int dir in new[] { -1, 1 })
                    {
                        var p = new Vector2(x, z);
                        for (int i = 0; i < 26; i++)
                        {
                            var grad = Grad(p.x, p.y);
                            if (grad.sqrMagnitude < 1e-6f)
                                break;
                            var along = new Vector2(-grad.y, grad.x).normalized * dir;
                            p += along * step;
                            if (Mathf.Abs(p.x) > half - 8f || Mathf.Abs(p.y) > half - 8f)
                                break;
                            if (SlopeDegAt(p.x, p.y) < slopeOff)
                                break;
                            run.Add(p);
                        }
                    }

                    // Lone steep cells are noise; two chained pieces make a wall.
                    if (run.Count < 2)
                        continue;
                    foreach (var p in run)
                        Emit(p.x, p.y, fam);
                }
            }
        }

        /// <summary>
        /// Find the island's most prominent peaks and hand each a scaled
        /// MountainRange silhouette — the black spiky skyline the packs use
        /// in their own promo scenes. Variant follows prominence order.
        /// </summary>
        private void ComputeSummits()
        {
            _summits.Clear();
            if (_dials.SummitCount <= 0)
                return;   // e.g. Canyonlands: mesas, no alpine ranges
            var rng = new System.Random(settings.Seed * 83 + 7);
            float floor = settings.SeaLevel + 55f;
            float half = settings.WorldSize * 0.5f;

            var peaks = new List<(float h, int gx, int gz)>();
            for (int gz = 4; gz < _heightRes - 4; gz += 2)
            {
                for (int gx = 4; gx < _heightRes - 4; gx += 2)
                {
                    float h = _heightGrid[gx, gz];
                    if (h < floor)
                        continue;
                    bool isMax = true;
                    for (int dz = -4; dz <= 4 && isMax; dz += 2)
                        for (int dx = -4; dx <= 4 && isMax; dx += 2)
                            if (_heightGrid[gx + dx, gz + dz] > h)
                                isMax = false;
                    if (isMax)
                        peaks.Add((h, gx, gz));
                }
            }
            peaks.Sort((a, b) => b.h.CompareTo(a.h));

            foreach (var (h, gx, gz) in peaks)
            {
                if (_summits.Count >= _dials.SummitCount)
                    break;
                float x = gx * _heightCell - half;
                float z = gz * _heightCell - half;

                bool tooClose = false;
                foreach (var s in _summits)
                {
                    float dx = x - s.Pos.x, dz = z - s.Pos.z;
                    if (dx * dx + dz * dz < 170f * 170f)
                    {
                        tooClose = true;
                        break;
                    }
                }
                if (tooClose)
                    continue;

                // Ridge axis: the direction whose flanks stay highest.
                float bestYaw = 0f, bestScore = float.MinValue;
                for (int a = 0; a < 12; a++)
                {
                    float ang = a * Mathf.PI / 12f;
                    float dx = Mathf.Sin(ang), dz = Mathf.Cos(ang);
                    float score = SampleHeight(x + dx * 35f, z + dz * 35f)
                                  + SampleHeight(x - dx * 35f, z - dz * 35f);
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestYaw = ang * Mathf.Rad2Deg;
                    }
                }

                float scale = Mathf.Clamp((h - settings.SeaLevel) / 16f, 4.5f, 7f);
                _summits.Add(new SummitSite
                {
                    Pos = new Vector3(x, h, z),
                    YawDeg = bestYaw + ((float)rng.NextDouble() - 0.5f) * 10f,
                    Scale = scale,
                    Variant = _summits.Count,
                });

                // Seat the range in terrain: raise a shoulder pedestal under
                // its footprint so the single-sided shell never shows its
                // hollow underside where the ground falls away.
                RaiseSummitPedestal(x, z, h, scale);
            }
        }

        /// <summary>
        /// Raise the height grid to a gentle dome at the summit mesh's base
        /// height across its footprint, blending back to natural terrain
        /// beyond. The blend ring comes out steep, which the cliff tracer
        /// (running after) dresses in crags — a natural rocky shoulder.
        /// </summary>
        private void RaiseSummitPedestal(float x, float z, float summitH, float scale)
        {
            float half = settings.WorldSize * 0.5f;
            float baseY = summitH - scale * 2.2f + 1.5f;   // matches WonderPlacer sink
            float radius = 14f * scale;                    // covers the full mesh skirt
            const float blend = 85f;                       // long, gentle shoulder
            int reach = Mathf.CeilToInt((radius * 1.2f + blend) / _heightCell);
            int cgx = Mathf.RoundToInt((x + half) / _heightCell);
            int cgz = Mathf.RoundToInt((z + half) / _heightCell);

            for (int dz = -reach; dz <= reach; dz++)
            {
                for (int dx = -reach; dx <= reach; dx++)
                {
                    int gx = cgx + dx, gz = cgz + dz;
                    if (gx < 0 || gz < 0 || gx >= _heightRes || gz >= _heightRes)
                        continue;
                    float d = Mathf.Sqrt(dx * dx + dz * dz) * _heightCell;
                    // A noisy rim: the pedestal edge wanders so the mountain
                    // base never reads as a clean circle / straight seam.
                    float rimNoise = 0.86f + 0.28f * Mathf.PerlinNoise(
                        (x + dx * _heightCell) * 0.013f + 91f,
                        (z + dz * _heightCell) * 0.013f - 47f);
                    float effRadius = radius * rimNoise;
                    if (d > effRadius + blend)
                        continue;
                    float target;
                    if (d <= effRadius)
                        target = baseY - (d / effRadius) * 2.5f;
                    else
                        target = Mathf.Lerp(baseY - 2.5f, _heightGrid[gx, gz],
                            Mathf.SmoothStep(0f, 1f, (d - effRadius) / blend));
                    if (target > _heightGrid[gx, gz])
                        _heightGrid[gx, gz] = target;
                }
            }
        }

        // ---- The volcano (VolcanoIsle profile) ----

        private Vector2 _volcanoPos;
        private float _volcanoR, _volcanoPeakY, _craterFloorY;
        private bool _hasVolcano;
        public bool HasVolcano => _hasVolcano;
        public Vector3 VolcanoCrater =>
            new(_volcanoPos.x, _craterFloorY, _volcanoPos.y);
        public float VolcanoRadius => _volcanoR;
        public float VolcanoRimY => _volcanoPeakY;

        /// <summary>0 in the green lowlands, 1 on the ash cone.</summary>
        public float VolcanoAsh01(float x, float z)
        {
            if (!_hasVolcano)
                return 0f;
            float dx = x - _volcanoPos.x, dz = z - _volcanoPos.y;
            float d = Mathf.Sqrt(dx * dx + dz * dz);
            return Mathf.SmoothStep(1f, 0f, d / (_volcanoR * 1.08f));
        }

        /// <summary>
        /// Stamp the cone and crater into the height grid. Sited away from
        /// the rivers so no watercourse crosses the lava country.
        /// </summary>
        private void ComputeVolcano()
        {
            _hasVolcano = false;
            if (!_dials.Volcano)
                return;

            var rng = new System.Random(settings.Seed * 173 + 29);
            float half = settings.WorldSize * 0.5f;
            const float R = 185f;
            for (int attempt = 0; attempt < 60 && !_hasVolcano; attempt++)
            {
                float ang = (float)rng.NextDouble() * Mathf.PI * 2f;
                float dist = 90f + (float)rng.NextDouble() * 130f;
                var p = new Vector2(Mathf.Cos(ang), Mathf.Sin(ang)) * dist;
                if (SampleHeight(p.x, p.y) - settings.SeaLevel < 8f)
                    continue;
                bool nearRiver = false;
                foreach (var river in _rivers)
                {
                    foreach (var rp in river)
                    {
                        float dx = rp.x - p.x, dz = rp.z - p.y;
                        if (dx * dx + dz * dz < 110f * 110f) { nearRiver = true; break; }
                    }
                    if (nearRiver) break;
                }
                if (nearRiver)
                    continue;
                _volcanoPos = p;
                _hasVolcano = true;
            }
            if (!_hasVolcano)
            {
                // Fallback: island center, rivers or not.
                _volcanoPos = Vector2.zero;
                _hasVolcano = true;
            }

            _volcanoR = R;
            _volcanoPeakY = settings.SeaLevel + 102f;
            _craterFloorY = _volcanoPeakY - 24f;
            const float craterR = 34f;

            int reach = Mathf.CeilToInt(R / _heightCell) + 2;
            int cgx = Mathf.RoundToInt((_volcanoPos.x + half) / _heightCell);
            int cgz = Mathf.RoundToInt((_volcanoPos.y + half) / _heightCell);
            for (int dz = -reach; dz <= reach; dz++)
            {
                for (int dx = -reach; dx <= reach; dx++)
                {
                    int gx = cgx + dx, gz = cgz + dz;
                    if (gx < 0 || gz < 0 || gx >= _heightRes || gz >= _heightRes)
                        continue;
                    float d = Mathf.Sqrt(dx * dx + dz * dz) * _heightCell;
                    if (d > R)
                        continue;

                    if (d < craterR)
                    {
                        // Inside the crater: bowl from rim down to the floor.
                        float t = Mathf.SmoothStep(0f, 1f, d / craterR);
                        _heightGrid[gx, gz] = Mathf.Lerp(_craterFloorY, _volcanoPeakY, t);
                    }
                    else
                    {
                        // Concave flank: steep near the rim, easing to the base.
                        float t = Mathf.InverseLerp(craterR, R, d);
                        float cone = _volcanoPeakY
                                     - Mathf.Pow(t, 1.35f) * (_volcanoPeakY - settings.SeaLevel - 6f);
                        if (cone > _heightGrid[gx, gz])
                            _heightGrid[gx, gz] = cone;
                    }
                }
            }
        }

        /// <summary>Max of badlands rockiness and local crag fields — drives
        /// rock scatter and large cliff-mesh placement.</summary>
        public float SampleCragginess01(float x, float z)
        {
            float best = SampleRockiness01(x, z);
            foreach (var seed in _outcropSeeds)
            {
                float dx = x - seed.Pos.x, dz = z - seed.Pos.y;
                float d = Mathf.Sqrt(dx * dx + dz * dz);
                if (d < seed.Radius)
                    best = Mathf.Max(best, Mathf.SmoothStep(0f, 1f, 1f - d / seed.Radius) * seed.Strength);
            }
            return Mathf.Clamp01(best);
        }

        // ---- Sim-facing resource truth layers -------------------------
        // Pure derived reads for the ecosystem sim: where food, shelter,
        // and drinking water actually are. No caching — every input is
        // already a cheap baked-grid or stamp lookup.

        /// <summary>
        /// Herbivore food availability. Richest at forest edges and moist
        /// open meadows; suppressed by cold, bare rock, fresh burn scars,
        /// and volcanic ash.
        /// </summary>
        public float SampleBrowse01(float x, float z)
        {
            float h = SampleHeight(x, z) - settings.SeaLevel;
            if (h < 0.4f)
                return 0f;
            float density = ForestDensity01(x, z);
            float edge = density * (1f - density) * 4f;        // peaks mid-density
            float meadow = (1f - density) * Mathf.Clamp01(SampleMoisture(x, z) * 1.5f);
            float warm = Mathf.SmoothStep(0f, 1f,
                Mathf.InverseLerp(0.10f, 0.32f, SampleTemperature01(x, z)));
            float fertile = (1f - SampleRockiness01(x, z) * 0.75f)
                            * (1f - SampleCragginess01(x, z) * 0.85f);
            float healthy = (1f - BurnScar01(x, z) * 0.85f)
                            * (1f - VolcanoAsh01(x, z));
            return Mathf.Clamp01(Mathf.Max(edge, meadow) * warm * fertile * healthy);
        }

        /// <summary>
        /// Concealment from predators (and weather): forest cores, crag
        /// fields, marsh reeds. Burnt forest hides no one.
        /// </summary>
        public float SampleCover01(float x, float z)
        {
            float forest = ForestDensity01(x, z) * (1f - BurnScar01(x, z) * 0.9f);
            float rocks = SampleCragginess01(x, z) * 0.65f + SampleRockiness01(x, z) * 0.25f;
            float marsh = SampleBiome(x, z) == Biome.Wetland ? 0.45f : 0f;
            return Mathf.Clamp01(Mathf.Max(forest, Mathf.Max(rocks, marsh)));
        }

        /// <summary>Drinking-water reachability: 1 at a fresh shore, ~0 beyond a few hundred meters.</summary>
        public float SampleWaterAccess01(float x, float z)
        {
            return Mathf.Exp(-SampleFreshwaterDistance(x, z) / 140f);
        }

        // ---- Rivers ----

        /// <summary>
        /// D8 flow accumulation on a coarse pre-river heightfield: every
        /// cell drains to its steepest-descent neighbor, and drainage area
        /// accumulates downstream. River sources are the well-separated
        /// cells where real catchments first concentrate — ravine heads,
        /// not random spots on the map.
        /// </summary>
        private List<Vector2> ComputeFlowSources()
        {
            const int res = 128;
            float half = settings.WorldSize * 0.5f;
            float cell = settings.WorldSize / (res - 1);

            var height = new float[res * res];
            for (int gz = 0; gz < res; gz++)
                for (int gx = 0; gx < res; gx++)
                    height[gz * res + gx] = RawHeight(gx * cell - half, gz * cell - half);

            // Downstream neighbor per cell (D8 steepest descent).
            var downstream = new int[res * res];
            for (int gz = 0; gz < res; gz++)
            {
                for (int gx = 0; gx < res; gx++)
                {
                    int i = gz * res + gx;
                    downstream[i] = -1;
                    float best = height[i];
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            if (dx == 0 && dz == 0) continue;
                            int nx = gx + dx, nz = gz + dz;
                            if (nx < 0 || nz < 0 || nx >= res || nz >= res) continue;
                            // Diagonal steps are longer — compare true slope.
                            float nh = height[nz * res + nx];
                            float drop = (height[i] - nh) / ((dx != 0 && dz != 0) ? 1.414f : 1f);
                            if (height[i] - drop < best)
                            {
                                best = height[i] - drop;
                                downstream[i] = nz * res + nx;
                            }
                        }
                    }
                }
            }

            // Accumulate drainage area from the top down.
            var order = new int[res * res];
            for (int i = 0; i < order.Length; i++) order[i] = i;
            System.Array.Sort(order, (a, b) => height[b].CompareTo(height[a]));
            var accum = new float[res * res];
            for (int i = 0; i < accum.Length; i++) accum[i] = 1f;
            foreach (int i in order)
                if (downstream[i] >= 0)
                    accum[downstream[i]] += accum[i];

            // Candidate sources: cells where a catchment has concentrated,
            // high enough to march from, ranked by drainage area and kept
            // well apart so each river owns its valley.
            float minH = 30f * _dials.HeightMul;
            var candidates = new List<int>();
            for (int i = 0; i < accum.Length; i++)
                if (accum[i] >= 25f && height[i] >= minH)
                    candidates.Add(i);
            candidates.Sort((a, b) => accum[b].CompareTo(accum[a]));

            // A candidate must actually drain to the sea. Sources inside
            // closed basins (summit bowls, karst pits) would carve escape
            // trenches through the rim and flood the bowl into a lake,
            // drowning whatever lived there.
            bool DrainsToSea(int start)
            {
                int i = start;
                for (int steps = 0; steps < res * res; steps++)
                {
                    if (height[i] < settings.SeaLevel + 0.5f)
                        return true;
                    int next = downstream[i];
                    if (next < 0)
                        return false;                       // interior pit
                    i = next;
                }
                return false;
            }

            var picked = new List<Vector2>();
            float minSepSq = 130f * 130f;
            foreach (int i in candidates)
            {
                if (!DrainsToSea(i))
                    continue;
                var p = new Vector2((i % res) * cell - half, (i / res) * cell - half);
                bool clear = true;
                foreach (var q in picked)
                    if ((p - q).sqrMagnitude < minSepSq) { clear = false; break; }
                if (!clear)
                    continue;
                picked.Add(p);
                if (picked.Count >= _dials.RiverCount * 3)
                    break;
            }
            return picked;
        }

        private void BuildRivers()
        {
            _rivers.Clear();
            _riverGrid.Clear();
            if (_dials.RiverCount <= 0)
                return;

            var rng = new System.Random(settings.Seed * 31 + 5);
            float half = settings.WorldSize * 0.5f;
            int attempts = 0;

            // Flow-derived sourcing: rivers rise where the terrain says
            // water concentrates, not at random high points.
            var flowSources = settings.FlowDerivedRivers ? ComputeFlowSources() : null;
            int nextSource = 0;

            while (_rivers.Count < _dials.RiverCount && attempts++ < 60)
            {
                float sx, sz;
                if (flowSources != null && nextSource < flowSources.Count)
                {
                    var s = flowSources[nextSource++];
                    sx = s.x;
                    sz = s.y;
                }
                else
                {
                    sx = ((float)rng.NextDouble() * 2f - 1f) * half * 0.7f;
                    sz = ((float)rng.NextDouble() * 2f - 1f) * half * 0.7f;
                }
                if (RawHeight(sx, sz) < 30f * _dials.HeightMul)
                    continue;

                var path = MarchRiver(sx, sz, rng);
                if (path == null || path.Count < 30)
                    continue;

                // Bed heights: monotonically descending along the path so the
                // channel never runs uphill (this is what kills dry patches —
                // the carve digs THROUGH bumps instead of following them).
                float running = float.MaxValue;
                for (int i = 0; i < path.Count; i++)
                {
                    var p = path[i];
                    running = Mathf.Min(running, RawHeight(p.x, p.z) - settings.RiverDepth);
                    p.y = running;
                    path[i] = p;
                }

                // Estuary: the final reach flares out where it meets the sea.
                int tail = 0;
                for (int i = path.Count - 1; i >= 0 && path[i].y < settings.SeaLevel + 1.5f; i--)
                    tail++;
                for (int k = 0; k < tail; k++)
                {
                    int i = path.Count - 1 - k;
                    var p = path[i];
                    p.w *= Mathf.Lerp(2.4f, 1f, k / Mathf.Max(tail - 1f, 1f));
                    path[i] = p;
                }

                _rivers.Add(path);

            }

            // Tributary confluence: where two rivers meet, the downstream
            // trunk swells (width adds in quadrature, like drainage area).
            for (int a = 0; a < _rivers.Count; a++)
            {
                for (int b = 0; b < _rivers.Count; b++)
                {
                    if (a == b) continue;
                    var pathA = _rivers[a];
                    int junction = -1;
                    foreach (var pb in _rivers[b])
                    {
                        for (int i = 0; i < pathA.Count; i++)
                        {
                            float dx = pathA[i].x - pb.x, dz = pathA[i].z - pb.z;
                            if (dx * dx + dz * dz < 64f) { junction = i; break; }
                        }
                        if (junction >= 0) break;
                    }
                    if (junction >= 0)
                    {
                        for (int i = junction; i < pathA.Count; i++)
                        {
                            var p = pathA[i];
                            p.w = Mathf.Min(Mathf.Sqrt(p.w * p.w * 1.9f), settings.RiverMaxWidth * 2.2f);
                            pathA[i] = p;
                        }
                    }
                }
            }

            // THE canyon: the river with the highest source cuts a deep,
            // steep-walled canyon through its mid-run.
            _gorgeRiver = -1;
            float bestSource = 24f;
            for (int i = 0; i < _rivers.Count; i++)
            {
                float src = RawHeight(_rivers[i][0].x, _rivers[i][0].z);
                if (src > bestSource) { bestSource = src; _gorgeRiver = i; }
            }
            if (_gorgeRiver >= 0)
            {
                var path = _rivers[_gorgeRiver];
                for (int i = 0; i < path.Count; i++)
                {
                    float t = i / (float)(path.Count - 1);
                    float gorge = Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(0.22f, 0.42f, t))
                                * Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(0.8f, 0.6f, t));
                    if (gorge > 0f)
                    {
                        var p = path[i];
                        p.y -= gorge * _dials.CanyonDepth;
                        p.w *= 1f - gorge * 0.4f;   // narrow the slot so the
                                                    // ribbon stays off the walls
                        path[i] = p;
                    }
                }

                // Re-enforce monotonic descent — the carve taper would let
                // the bed climb back OUT of the canyon (water uphill). The
                // canyon instead shallows naturally as terrain descends to
                // meet the bed. Floor above sea level so it never becomes a
                // flooded fjord.
                float runMin = float.MaxValue;
                for (int i = 0; i < path.Count; i++)
                {
                    var p = path[i];
                    runMin = Mathf.Min(runMin, p.y);
                    p.y = Mathf.Max(runMin, settings.SeaLevel + 0.8f);
                    path[i] = p;
                }
            }

            PlaceFordsAndCrossing();

            // Hash points plus segment midpoints for gap-free influence.
            foreach (var path in _rivers)
            {
                for (int i = 0; i < path.Count; i++)
                {
                    AddRiverPointToGrid(path[i]);
                    if (i + 1 < path.Count)
                        AddRiverPointToGrid(Vector4.Lerp(path[i], path[i + 1], 0.5f));
                }
            }
        }

        private void AddRiverPointToGrid(Vector4 p)
        {
            long key = CellKey(p.x, p.z);
            if (!_riverGrid.TryGetValue(key, out var list))
                _riverGrid[key] = list = new List<Vector4>();
            list.Add(p);
        }

        /// <summary>
        /// Keep the land traversable: every river gets shallow fords (the
        /// terrain floor rises to a gravel bar just under the unchanged
        /// water surface), and the canyon gets one crossing where the rims
        /// are lowest — a talus ramp down either wall plus a center ford.
        /// </summary>
        private void PlaceFordsAndCrossing()
        {
            _fords.Clear();
            _gorgeCross = default;
            if (!settings.EnsureConnectivity)
                return;

            var rng = new System.Random(settings.Seed * 503 + 11);
            for (int r = 0; r < _rivers.Count; r++)
            {
                var path = _rivers[r];
                // The gorge river only fords above its canyon window — the
                // canyon itself is crossed at the ramp below.
                var stops = r == _gorgeRiver
                    ? new[] { 0.12f }
                    : new[] { 0.30f, 0.66f };
                foreach (float tBase in stops)
                {
                    float t = Mathf.Clamp01(tBase + ((float)rng.NextDouble() - 0.5f) * 0.10f);
                    int i = Mathf.Clamp(Mathf.RoundToInt(t * (path.Count - 1)), 0, path.Count - 1);
                    var p = path[i];
                    if (p.y < settings.SeaLevel + 1f)          // estuary reach — skip
                        continue;
                    _fords.Add(new FordSite
                    {
                        Pos = new Vector2(p.x, p.z),
                        Radius = Mathf.Max(p.w * 1.6f, 10f),
                    });
                }
            }

            // Canyon crossing: scan the gorge's mid-run for the spot where
            // the walls are lowest, like a real trailhead saddle.
            if (_gorgeRiver < 0)
                return;
            var gp = _rivers[_gorgeRiver];
            int bestI = -1;
            float bestWall = float.MaxValue, bestRimA = 0f, bestRimB = 0f;
            var bestPerp = Vector2.zero;
            for (int i = (int)(gp.Count * 0.30f); i < (int)(gp.Count * 0.72f); i++)
            {
                var p = gp[i];
                var q = gp[Mathf.Min(i + 1, gp.Count - 1)];
                var fdir = new Vector2(q.x - p.x, q.z - p.z).normalized;
                var perp = new Vector2(-fdir.y, fdir.x);
                float sampleD = p.w * 2f + 26f;
                float rimA = RawHeight(p.x - perp.x * sampleD, p.z - perp.y * sampleD);
                float rimB = RawHeight(p.x + perp.x * sampleD, p.z + perp.y * sampleD);
                float wall = Mathf.Max(rimA, rimB) - p.y;
                if (wall < 6f)
                    continue;                                  // shallow reach — crossable anyway
                if (wall < bestWall)
                {
                    bestWall = wall; bestI = i; bestPerp = perp;
                    bestRimA = rimA; bestRimB = rimB;
                }
            }
            if (bestI < 0)
                return;
            {
                var p = gp[bestI];
                var q = gp[Mathf.Min(bestI + 1, gp.Count - 1)];
                _gorgeCross = new GorgeCrossing
                {
                    Active = true,
                    Pos = new Vector2(p.x, p.z),
                    Dir = bestPerp,
                    RiverDir = new Vector2(q.x - p.x, q.z - p.z).normalized,
                    BedY = p.y,
                    RimA = bestRimA,
                    RimB = bestRimB,
                    HalfLen = Mathf.Clamp(bestWall * 1.7f + p.w, 30f, 130f),
                    HalfWidth = Mathf.Max(p.w * 1.1f, 11f),
                };
                _fords.Add(new FordSite
                {
                    Pos = _gorgeCross.Pos,
                    Radius = Mathf.Max(p.w * 1.7f, 12f),
                });
            }
        }

        private List<Vector4> MarchRiver(float x, float z, System.Random rng)
        {
            var path = new List<Vector4>();
            var momentum = Vector2.zero;
            float half = settings.WorldSize * 0.5f;
            float traveled = 0f;
            const float step = 5f;
            float meanderPhase = (float)rng.NextDouble() * Mathf.PI * 2f;

            for (int i = 0; i < 900; i++)
            {
                float h = RawHeight(x, z);
                if (h <= settings.SeaLevel + 0.3f)
                    return path;                       // reached the sea — done
                if (Mathf.Abs(x) > half - 10f || Mathf.Abs(z) > half - 10f)
                    return path;

                float width = Mathf.Lerp(settings.RiverMinWidth, settings.RiverMaxWidth,
                    Mathf.Clamp01(traveled / 500f));
                path.Add(new Vector4(x, 0f, z, width));

                // Steepest descent with momentum (momentum carries the river
                // across small flats and keeps bends smooth).
                const float e = 4f;
                var grad = new Vector2(
                    RawHeight(x + e, z) - RawHeight(x - e, z),
                    RawHeight(x, z + e) - RawHeight(x, z - e));
                var downhill = -grad;
                if (downhill.sqrMagnitude < 1e-6f)
                    downhill = momentum.sqrMagnitude > 1e-6f
                        ? momentum
                        : new Vector2((float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f);

                // Lowland meandering: where the gradient is weak, gravity
                // stops dictating and the channel snakes — the signature of
                // a mature river. Steep ground stays gradient-driven.
                float slopeMag = grad.magnitude / (2f * e);
                float meander = Mathf.Clamp01(1f - slopeMag / 0.05f);
                var target = downhill.normalized;
                if (meander > 0.01f)
                {
                    meanderPhase += step * 0.05f;
                    var side = new Vector2(-momentum.y, momentum.x);
                    target = (target + side * (Mathf.Sin(meanderPhase) * meander * 0.85f)).normalized;
                }

                momentum = (momentum * 0.65f + target * 0.35f).normalized;
                x += momentum.x * step;
                z += momentum.y * step;
                traveled += step;
            }
            return null;                                // never found the sea — discard
        }

        private static long CellKey(float x, float z)
        {
            int cx = Mathf.FloorToInt(x / RiverCellSize);
            int cz = Mathf.FloorToInt(z / RiverCellSize);
            return ((long)cx << 32) ^ (uint)cz;
        }

        /// <summary>0 = untouched, 1 = center of a river channel.</summary>
        public float RiverInfluence01(float x, float z)
        {
            GetRiverData(x, z, out float influence, out _);
            return influence;
        }

        /// <summary>Influence plus the target bed height of the nearest channel point.</summary>
        private void GetRiverData(float x, float z, out float influence, out float bed)
        {
            influence = 0f;
            bed = 0f;
            if (_riverGrid.Count == 0)
                return;

            int cx = Mathf.FloorToInt(x / RiverCellSize);
            int cz = Mathf.FloorToInt(z / RiverCellSize);
            for (int dz = -1; dz <= 1; dz++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    long key = ((long)(cx + dx) << 32) ^ (uint)(cz + dz);
                    if (!_riverGrid.TryGetValue(key, out var points))
                        continue;
                    foreach (var p in points)
                    {
                        float ddx = p.x - x, ddz = p.z - z;
                        float d = Mathf.Sqrt(ddx * ddx + ddz * ddz);
                        // 1 inside the inner quarter, fading to 0 at full width.
                        float t = Mathf.InverseLerp(p.w, p.w * 0.25f, d);
                        if (t > influence)
                        {
                            influence = t;
                            bed = p.y;
                        }
                    }
                }
            }
            influence = Mathf.Clamp01(influence);
        }

        /// <summary>Build river ribbons + pond discs of translucent water.</summary>
        public void BuildRiverWater()
        {
            if (waterMaterial == null)
                return;

            // Lake surfaces: one merged quad-per-cell mesh at each lake's
            // priority-flood water level (ponds are carved basins the flood
            // detects, so all standing water renders through this one path).
            float half = settings.WorldSize * 0.5f;
            foreach (var lake in _lakes)
            {
                // Emit the flooded cells PLUS a one-cell ring: the mesh
                // border then lands inside the rising bank, so the visible
                // shoreline is the smooth terrain/water intersection line,
                // not the hovering stair-step edge of the quad grid.
                var cells = new HashSet<(int, int)>();
                foreach (var (cx, cz) in lake.Cells)
                {
                    cells.Add((cx, cz));
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int nx = cx + dx, nz = cz + dz;
                            if (nx < 0 || nz < 0 || nx >= _heightRes || nz >= _heightRes)
                                continue;
                            // Skirt only onto rising banks — dilating over a
                            // cliff edge leaves water hanging in the air.
                            if (_heightGrid[nx, nz] >= lake.WaterLevel - 0.4f)
                                cells.Add((nx, nz));
                        }
                    }
                }

                var verts = new List<Vector3>(cells.Count * 4);
                var tris = new List<int>(cells.Count * 6);
                float y = lake.WaterLevel + 0.06f;
                foreach (var (gx, gz) in cells)
                {
                    float x0 = gx * _heightCell - half - _heightCell * 0.5f;
                    float z0 = gz * _heightCell - half - _heightCell * 0.5f;
                    int b = verts.Count;
                    verts.Add(new Vector3(x0, y, z0));
                    verts.Add(new Vector3(x0 + _heightCell, y, z0));
                    verts.Add(new Vector3(x0, y, z0 + _heightCell));
                    verts.Add(new Vector3(x0 + _heightCell, y, z0 + _heightCell));
                    tris.AddRange(new[] { b, b + 2, b + 1, b + 1, b + 2, b + 3 });
                }
                var mesh = new Mesh
                {
                    name = "LakeWater",
                    indexFormat = UnityEngine.Rendering.IndexFormat.UInt32,
                };
                mesh.SetVertices(verts);
                mesh.SetTriangles(tris, 0);
                mesh.RecalculateNormals();
                mesh.RecalculateBounds();
                var go = new GameObject("LakeWater") { hideFlags = HideFlags.DontSave };
                go.transform.SetParent(transform, false);
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
                go.AddComponent<MeshRenderer>().sharedMaterial =
                    lakeWaterMaterial != null ? lakeWaterMaterial : waterMaterial;
            }

            foreach (var rawRiver in _rivers)
            {
                // One pass of Chaikin corner-cutting so the ribbon flows in
                // curves instead of the march's straight segments. (Only the
                // mesh is smoothed — influence/bed queries keep the raw path.)
                var river = new List<Vector4>(rawRiver.Count * 2);
                river.Add(rawRiver[0]);
                for (int i = 0; i + 1 < rawRiver.Count; i++)
                {
                    river.Add(Vector4.Lerp(rawRiver[i], rawRiver[i + 1], 0.25f));
                    river.Add(Vector4.Lerp(rawRiver[i], rawRiver[i + 1], 0.75f));
                }
                river.Add(rawRiver[rawRiver.Count - 1]);

                var verts = new List<Vector3>();
                var uvs = new List<Vector2>();
                var tris = new List<int>();
                float along = 0f;

                for (int i = 0; i < river.Count; i++)
                {
                    var p = river[i];
                    var next = river[Mathf.Min(i + 1, river.Count - 1)];
                    var dir = new Vector2(next.x - p.x, next.z - p.z);
                    if (i > 0)
                        along += new Vector2(p.x - river[i - 1].x, p.z - river[i - 1].z).magnitude;
                    if (dir.sqrMagnitude < 1e-6f) dir = Vector2.up;
                    dir.Normalize();
                    var side = new Vector2(-dir.y, dir.x);

                    // p.y is the monotonically descending bed target, so the
                    // surface is continuous by construction.
                    float y = Mathf.Max(p.y + settings.RiverDepth * 0.55f, settings.SeaLevel + 0.02f);

                    float hw = p.w * 0.8f;
                    verts.Add(new Vector3(p.x + side.x * hw, y, p.z + side.y * hw));
                    verts.Add(new Vector3(p.x - side.x * hw, y, p.z - side.y * hw));
                    // V runs downstream so the shader can scroll flow along it.
                    float v = along * 0.06f;
                    uvs.Add(new Vector2(0f, v));
                    uvs.Add(new Vector2(1f, v));

                    if (i > 0)
                    {
                        int b = verts.Count - 4;
                        tris.AddRange(new[] { b, b + 2, b + 1, b + 1, b + 2, b + 3 });
                    }
                }

                var mesh = new Mesh { name = "RiverWater" };
                mesh.SetVertices(verts);
                mesh.SetUVs(0, uvs);
                mesh.SetTriangles(tris, 0);
                mesh.RecalculateNormals();
                mesh.RecalculateBounds();

                var go = new GameObject("RiverWater") { hideFlags = HideFlags.DontSave };
                go.transform.SetParent(transform, false);
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
                go.AddComponent<MeshRenderer>().sharedMaterial =
                    riverWaterMaterial != null ? riverWaterMaterial : waterMaterial;
            }
        }

        public Color SampleBiomeColor(float x, float z, float height)
        {
            float h = height - settings.SeaLevel;
            float moisture = SampleMoisture(x, z);
            float river = RiverInfluence01(x, z);

            // The old burn chars the ground toward soot; the volcano buries
            // its cone in dark ash.
            float scar = BurnScar01(x, z);
            float ash = VolcanoAsh01(x, z);
            if (scar > 0.03f || ash > 0.03f)
            {
                var baseCol = SampleBiomeColorUnscarred(x, z, height, h, moisture, river);
                baseCol = Color.Lerp(baseCol, new Color(0.16f, 0.14f, 0.12f), scar * 0.62f);
                return Color.Lerp(baseCol, new Color(0.18f, 0.16f, 0.15f), ash * 0.9f);
            }
            return SampleBiomeColorUnscarred(x, z, height, h, moisture, river);
        }

        private Color SampleBiomeColorUnscarred(float x, float z, float height,
            float h, float moisture, float river)
        {

            // Canyonlands: every steep mesa step face is banded red rock,
            // river or no river.
            if (_dials.MesaTerraces && h > 3f)
            {
                const float e = 2f;
                float sgx = SampleHeight(x + e, z) - SampleHeight(x - e, z);
                float sgz = SampleHeight(x, z + e) - SampleHeight(x, z - e);
                float faceSlope = Mathf.Atan(Mathf.Sqrt(sgx * sgx + sgz * sgz) / (2f * e)) * Mathf.Rad2Deg;
                if (faceSlope > 30f)
                {
                    int mband = Mathf.FloorToInt(height / 3.5f);
                    float mn = (Mathf.PerlinNoise(x * 0.11f, z * 0.11f) - 0.5f) * 0.10f;
                    var mc = (mband & 1) == 0 ? settings.CanyonRockDark : settings.CanyonRockLight;
                    return new Color(
                        Mathf.Clamp01(mc.r + mn),
                        Mathf.Clamp01(mc.g + mn * 0.8f),
                        Mathf.Clamp01(mc.b + mn * 0.6f));
                }
            }

            // River channels read as wet mud banks (blended, not a hard cut).
            if (river > 0.25f && h > 0f)
            {
                // Canyon walls read as banded red strata, not mud.
                GetRiverData(x, z, out _, out float bed);
                float wall = height - bed;
                if (wall > 9f)
                {
                    int band = Mathf.FloorToInt(wall / 3.5f);
                    float n = (Mathf.PerlinNoise(x * 0.11f, z * 0.11f) - 0.5f) * 0.10f;
                    var c = (band & 1) == 0 ? settings.CanyonRockDark : settings.CanyonRockLight;
                    return new Color(
                        Mathf.Clamp01(c.r + n),
                        Mathf.Clamp01(c.g + n * 0.8f),
                        Mathf.Clamp01(c.b + n * 0.6f));
                }
                var bank = h < settings.SandBandTop ? settings.Sand : settings.GrassLush;
                return Color.Lerp(bank, settings.RiverMud, Mathf.InverseLerp(0.25f, 0.8f, river));
            }

            switch (SampleBiome(x, z))
            {
                case Biome.Ocean:
                    return Color.Lerp(settings.DeepWaterBed, settings.Sand, Mathf.Clamp01(1f + h / 12f) * 0.5f);
                case Biome.Beach:
                    return settings.Sand;
                case Biome.Wetland:
                    return Color.Lerp(settings.Wetland, settings.GrassLush, moisture * 0.3f);
                case Biome.ForestFloor:
                {
                    float core = ForestDensity01(x, z);
                    return Color.Lerp(settings.GrassLush, settings.Forest, 0.35f + core * 0.5f);
                }
                case Biome.Scrub:
                    return Color.Lerp(settings.Scrub, settings.GrassDry, moisture);
                case Biome.AlpineMeadow:
                    return Color.Lerp(settings.AlpineMeadow, settings.Rock, Mathf.InverseLerp(
                        settings.GrassBandTop, settings.RockBandTop, h) * 0.4f);
                case Biome.Rock:
                    return settings.Rock;
                case Biome.Snow:
                    return settings.Snow;
                default: // Meadow
                {
                    var grass = Color.Lerp(settings.GrassDry, settings.GrassLush, moisture);
                    float toward = Mathf.InverseLerp(settings.GrassBandTop, settings.RockBandTop, h);
                    return Color.Lerp(grass, settings.Rock, toward * 0.35f);
                }
            }
        }

        /// <summary>Large-scale brightness variation breaking up uniformity.</summary>
        private float MacroVariation(float x, float z)
        {
            return 0.9f + _moistureNoise.Sample(x + 511f, z - 733f, 1f / 90f) * 0.2f;
        }

        // ---- Mesh building ----

        private void BuildChunk(int cx, int cz, Vector3 origin)
        {
            int quads = Mathf.RoundToInt(settings.ChunkSize / settings.QuadSize);
            int triCount = quads * quads * 2;

            var vertices = new List<Vector3>(triCount * 3);
            var colors = new List<Color>(triCount * 3);
            var splats = new List<Vector4>(triCount * 3);
            var indices = new List<int>(triCount * 3);

            for (int qz = 0; qz < quads; qz++)
            {
                for (int qx = 0; qx < quads; qx++)
                {
                    float x0 = origin.x + qx * settings.QuadSize;
                    float z0 = origin.z + qz * settings.QuadSize;
                    float x1 = x0 + settings.QuadSize;
                    float z1 = z0 + settings.QuadSize;

                    var p00 = new Vector3(x0, SampleHeight(x0, z0), z0);
                    var p10 = new Vector3(x1, SampleHeight(x1, z0), z0);
                    var p01 = new Vector3(x0, SampleHeight(x0, z1), z1);
                    var p11 = new Vector3(x1, SampleHeight(x1, z1), z1);

                    // Alternate the quad split direction for a less uniform look.
                    bool flip = ((qx + qz) & 1) == 0;
                    if (flip)
                    {
                        AddTriangle(vertices, colors, splats, indices, p00, p01, p11);
                        AddTriangle(vertices, colors, splats, indices, p00, p11, p10);
                    }
                    else
                    {
                        AddTriangle(vertices, colors, splats, indices, p00, p01, p10);
                        AddTriangle(vertices, colors, splats, indices, p10, p01, p11);
                    }
                }
            }

            // Vertices are stored in world space; bake back to chunk-local.
            for (int i = 0; i < vertices.Count; i++)
                vertices[i] -= origin;

            var mesh = new Mesh
            {
                name = $"TerrainChunk_{cx}_{cz}",
                indexFormat = UnityEngine.Rendering.IndexFormat.UInt32,
            };
            mesh.SetVertices(vertices);
            mesh.SetColors(colors);
            mesh.SetUVs(1, splats);
            mesh.SetTriangles(indices, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            var go = new GameObject($"Chunk_{cx}_{cz}") { hideFlags = HideFlags.DontSave };
            go.transform.SetParent(transform, false);
            go.transform.position = origin;
            go.isStatic = true;
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            go.AddComponent<MeshRenderer>().sharedMaterial = terrainMaterial;
            go.AddComponent<MeshCollider>().sharedMesh = mesh;
        }

        /// <summary>Per-face color + splat weights sampled at the centroid.</summary>
        private void AddTriangle(List<Vector3> verts, List<Color> cols, List<Vector4> splats, List<int> tris,
            Vector3 a, Vector3 b, Vector3 c)
        {
            var centroid = (a + b + c) / 3f;
            // Ecotone blending: average four jittered samples so biome borders
            // hand off through mixed-color transition bands instead of a hard
            // enum switch on one face.
            var color = SampleBiomeColor(centroid.x, centroid.z, centroid.y) * 0.4f
                + SampleBiomeColor(centroid.x + 5f, centroid.z + 1.8f, centroid.y) * 0.2f
                + SampleBiomeColor(centroid.x - 3.4f, centroid.z + 4.2f, centroid.y) * 0.2f
                + SampleBiomeColor(centroid.x - 1.6f, centroid.z - 5f, centroid.y) * 0.2f;
            float variation = MacroVariation(centroid.x, centroid.z);
            color = new Color(color.r * variation, color.g * variation, color.b * variation, 1f);

            // Face slope from the actual triangle normal, so cliffs read as
            // rock regardless of altitude band.
            var faceNormal = Vector3.Cross(b - a, c - a).normalized;
            float slopeDeg = Vector3.Angle(faceNormal, Vector3.up);
            var splat = SampleSplatWeights(centroid.x, centroid.z, centroid.y, slopeDeg) * 0.4f
                + SampleSplatWeights(centroid.x + 5f, centroid.z + 1.8f, centroid.y, slopeDeg) * 0.2f
                + SampleSplatWeights(centroid.x - 3.4f, centroid.z + 4.2f, centroid.y, slopeDeg) * 0.2f
                + SampleSplatWeights(centroid.x - 1.6f, centroid.z - 5f, centroid.y, slopeDeg) * 0.2f;

            int baseIndex = verts.Count;
            verts.Add(a); verts.Add(b); verts.Add(c);
            cols.Add(color); cols.Add(color); cols.Add(color);
            splats.Add(splat); splats.Add(splat); splats.Add(splat);
            tris.Add(baseIndex); tris.Add(baseIndex + 1); tris.Add(baseIndex + 2);
        }

        /// <summary>Texture blend weights: (grass, dirt, rock, snow).</summary>
        public Vector4 SampleSplatWeights(float x, float z, float height, float slopeDeg)
        {
            float h = height - settings.SeaLevel;
            float moisture = SampleMoisture(x, z);
            float river = RiverInfluence01(x, z);
            var biome = SampleBiome(x, z);

            // Slope-driven rock overrides everything — cliffs are rock at any height.
            float rockBySlope = Mathf.InverseLerp(30f, 44f, slopeDeg);
            float rockiness = SampleRockiness01(x, z);

            float snow = biome == Biome.Snow ? 1f
                : Mathf.InverseLerp(settings.RockBandTop, settings.RockBandTop + 9f, h) * 0.5f;
            float rock = Mathf.Max(Mathf.Max(rockBySlope, rockiness * 0.85f),
                biome == Biome.Rock ? 0.9f : 0f) * (1f - snow);

            float dirt = biome switch
            {
                Biome.Beach or Biome.Ocean => 1f,
                Biome.Scrub => 0.55f + Mathf.Pow(1f - moisture, 2f) * 0.3f,
                Biome.ForestFloor => 0.4f,      // leaf-litter floor under canopy
                Biome.Wetland => 0.5f,          // mud between reeds
                _ => Mathf.Pow(1f - moisture, 2.2f) * 0.4f,
            };
            dirt = Mathf.Max(dirt, river);
            dirt *= (1f - rock) * (1f - snow);

            // The burn is bare charred dirt; the cone is ash and rock.
            float scar = BurnScar01(x, z);
            if (scar > 0f)
                dirt = Mathf.Max(dirt, scar * 0.85f * (1f - rock) * (1f - snow));
            float ash = VolcanoAsh01(x, z);
            if (ash > 0f)
            {
                snow = 0f;
                dirt = Mathf.Max(dirt, ash * 0.7f * (1f - rock));
            }

            // Deep canyon walls are bare red rock — never snowbound, never
            // grassed over.
            if (river > 0.2f)
            {
                GetRiverData(x, z, out _, out float bed);
                if (height - bed > 9f)
                {
                    snow = 0f;
                    rock = Mathf.Max(rock, 0.75f);
                    dirt = Mathf.Min(dirt, 1f - rock);
                }
            }

            float grass = Mathf.Clamp01(1f - rock - snow - dirt);
            return new Vector4(grass, dirt, rock, snow);
        }

        private void ClearChunks()
        {
            for (int i = transform.childCount - 1; i >= 0; i--)
            {
                var child = transform.GetChild(i).gameObject;
                if (Application.isPlaying) Destroy(child);
                else DestroyImmediate(child);
            }
        }
    }
}
