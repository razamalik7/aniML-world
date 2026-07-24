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

        private float[,] _freshDistGrid;

        // Natural wonders: singular one-per-world features at computed sites.
        public struct WonderSite { public string Kind; public Vector3 Pos; public float YawDeg; public float Scale; }
        private readonly List<WonderSite> _wonders = new();
        public IReadOnlyList<WonderSite> Wonders => _wonders;

        // Baked heightfield: analytic height (with river/pond carves) is
        // rasterized once per generation, erosion runs on the grid, and all
        // queries become bilinear samples — faster AND erodible.
        private float[,] _heightGrid;
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

        public WorldSettings Settings => settings;

        public void Configure(WorldSettings s, Material terrainMat, Material waterMat)
        {
            settings = s;
            terrainMaterial = terrainMat;
            waterMaterial = waterMat;
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
            PlaceForests();
            PlaceOutcrops();
            TraceCoastCliffs();
            TraceInlandCliffs();
            ComputeSummits();
            ComputeWonders();
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

            // Waterfall: the largest single bed drop on any river.
            float bestDrop = 3.2f;
            Vector3 fallPos = default; float fallYaw = 0f; float fallScale = 1f;
            bool found = false;
            foreach (var river in _rivers)
            {
                for (int i = 0; i + 1 < river.Count; i++)
                {
                    float drop = river[i].y - river[i + 1].y;
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
                        found = true;
                    }
                }
            }
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
            if (!TryPlaceWonder(rng, "HotSprings", 240,
                    (x, z) => SampleBiome(x, z) == Biome.Scrub && SampleCragginess01(x, z) > 0.2f))
            {
                if (!TryPlaceWonder(rng, "HotSprings", 240, (x, z) => SampleBiome(x, z) == Biome.Scrub))
                    TryPlaceWonder(rng, "HotSprings", 240,
                        (x, z) => SampleCragginess01(x, z) > 0.4f && SampleBiome(x, z) != Biome.Ocean
                                  && SampleBiome(x, z) != Biome.Snow);
            }
            if (TryPlaceWonder(rng, "WhaleBones", 240,
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
            // Deep carves (the gorge) get sharpened influence → near-vertical
            // canyon walls instead of gentle banks.
            if (influence > 0f && raw - bed > 7f)
                influence = Mathf.Pow(influence, 0.45f);
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
                    if (lake.Cells.Count >= 60 && maxDepth >= 0.9f)
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

            while (_ponds.Count < 4 && attempts++ < 260)
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
            float baseHeight = Mathf.Pow(n, settings.HeightExponent) * settings.HeightAmplitude;

            // Ridged mountain layer, present only where the low-frequency
            // mountain mask says so — gives one coherent range, not spikes
            // everywhere.
            float ridgeSrc = _mountainNoise.Sample(w.x, w.y, 1f / settings.MountainFeatureSize);
            float ridge = 1f - Mathf.Abs(2f * ridgeSrc - 1f);      // fold: peaks at 0.5
            ridge = Mathf.Pow(ridge, 1.5f);                          // sharpen crests, keep broad shoulders
            float maskSrc = _mountainMaskNoise.Sample(w.x, w.y, 1f / settings.MountainMaskFeatureSize);
            float mountainMask = Mathf.SmoothStep(0f, 1f,
                Mathf.InverseLerp(settings.MountainMaskThreshold, settings.MountainMaskThreshold + 0.18f, maskSrc));
            float mountains = ridge * mountainMask * settings.MountainAmplitude;

            // Rocky badlands: jagged high-frequency relief inside a
            // low-frequency rockiness mask — breaks the everything-smooth
            // uniformity with genuinely rugged zones.
            float rockiness = RockinessFromMask(w.x, w.y);
            if (rockiness > 0f)
            {
                float roughSrc = _mountainNoise.Sample(w.x + 4096f, w.y - 2048f, 1f / settings.RockRoughFeatureSize);
                float rough = (1f - Mathf.Abs(2f * roughSrc - 1f)) * settings.RockRoughAmplitude;
                baseHeight += rough * rockiness;
            }

            // Island falloff on warped distance — an irregular coastline.
            float half = settings.WorldSize * 0.5f;
            float d = Mathf.Sqrt(w.x * w.x + w.y * w.y) / half;
            float islandMask = Mathf.Clamp01(1f - Mathf.Pow(d, settings.IslandFalloffPower));

            return (baseHeight + mountains) * islandMask - (1f - islandMask) * settings.EdgeSinkDepth;
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
                    float m = settings.BaseMoisture
                              + 0.62f * Mathf.Exp(-dist[gx, gz] / settings.MoistureDecayDistance)
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
                    float shadow = Mathf.Clamp01((maxUpwind - 18f) / 45f) * settings.RainShadowStrength;
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

            while (_forestSeeds.Count < settings.ForestCount && attempts++ < 400)
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

            // Ecotone tail: below the cutoff, a sparse "parkland" fringe of
            // lone trees rings each forest instead of a hard edge — the
            // in-between the hard cutoff was deleting. Far from any forest,
            // density is already 0 and plains stay truly open.
            if (density < settings.ForestDensityCutoff)
                return density * (density / settings.ForestDensityCutoff) * 0.18f;
            return density;
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
                return SampleFreshwaterDistance(x, z) < 55f ? Biome.Wetland : Biome.Meadow;
            }

            // Cold-shifted snowline: northern mountains hold snow lower.
            float snowLine = settings.RockBandTop + (temp - 0.4f) * 34f;
            if (h > snowLine)
                return Biome.Snow;
            if (h > settings.RockBandTop * 0.92f)
                return Biome.Rock;

            // Marshes are freshwater-fed lowlands, never salt coastline.
            if (SampleFreshwaterDistance(x, z) < 55f && h < 12f)
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
            const float slopeOn = 40f;   // slope (deg) that seeds a wall
            const float slopeOff = 32f;  // hysteresis: runs continue down to this
            const float step = 6.5f;     // meters between chained pieces

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
                float footY = SampleHeight(footX, footZ);
                if (footY < settings.SeaLevel + 0.5f)
                    return;

                _inlandCliffs.Add(new InlandCliff
                {
                    Pos = new Vector3(footX, footY - 0.6f, footZ),
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
                        || h - settings.SeaLevel > 46f)
                        fam = CliffFamily.Granite;
                    else if (biome == Biome.Scrub
                             || (SampleMoisture(x, z) < 0.36f && SampleTemperature01(x, z) > 0.5f))
                        fam = CliffFamily.RedRock;
                    else
                        fam = CliffFamily.Mossy;

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
                if (_summits.Count >= 3)
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

                _summits.Add(new SummitSite
                {
                    Pos = new Vector3(x, h, z),
                    YawDeg = bestYaw + ((float)rng.NextDouble() - 0.5f) * 10f,
                    Scale = Mathf.Clamp((h - settings.SeaLevel) / 14f, 5.5f, 9f),
                    Variant = _summits.Count,
                });
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

        // ---- Rivers ----

        private void BuildRivers()
        {
            _rivers.Clear();
            _riverGrid.Clear();
            if (settings.RiverCount <= 0)
                return;

            var rng = new System.Random(settings.Seed * 31 + 5);
            float half = settings.WorldSize * 0.5f;
            int attempts = 0;

            while (_rivers.Count < settings.RiverCount && attempts++ < 60)
            {
                float sx = ((float)rng.NextDouble() * 2f - 1f) * half * 0.7f;
                float sz = ((float)rng.NextDouble() * 2f - 1f) * half * 0.7f;
                if (RawHeight(sx, sz) < 30f)
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

            // THE gorge: the river with the highest source cuts a deep,
            // steep-walled canyon through its mid-run.
            int gorgeRiver = -1;
            float bestSource = 24f;
            for (int i = 0; i < _rivers.Count; i++)
            {
                float src = RawHeight(_rivers[i][0].x, _rivers[i][0].z);
                if (src > bestSource) { bestSource = src; gorgeRiver = i; }
            }
            if (gorgeRiver >= 0)
            {
                var path = _rivers[gorgeRiver];
                for (int i = 0; i < path.Count; i++)
                {
                    float t = i / (float)(path.Count - 1);
                    float gorge = Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(0.22f, 0.42f, t))
                                * Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(0.8f, 0.6f, t));
                    if (gorge > 0f)
                    {
                        var p = path[i];
                        p.y -= gorge * 9f;          // deepen the bed
                        p.w *= 1f - gorge * 0.25f;  // narrow the slot
                        path[i] = p;
                    }
                }
            }

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

        private List<Vector4> MarchRiver(float x, float z, System.Random rng)
        {
            var path = new List<Vector4>();
            var momentum = Vector2.zero;
            float half = settings.WorldSize * 0.5f;
            float traveled = 0f;
            const float step = 5f;

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

                momentum = (momentum * 0.65f + downhill.normalized * 0.35f).normalized;
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
                    for (int dz = -1; dz <= 1; dz++)
                        for (int dx = -1; dx <= 1; dx++)
                            cells.Add((cx + dx, cz + dz));

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
                go.AddComponent<MeshRenderer>().sharedMaterial = waterMaterial;
            }

            foreach (var river in _rivers)
            {
                var verts = new List<Vector3>();
                var tris = new List<int>();

                for (int i = 0; i < river.Count; i++)
                {
                    var p = river[i];
                    var next = river[Mathf.Min(i + 1, river.Count - 1)];
                    var dir = new Vector2(next.x - p.x, next.z - p.z);
                    if (dir.sqrMagnitude < 1e-6f) dir = Vector2.up;
                    dir.Normalize();
                    var side = new Vector2(-dir.y, dir.x);

                    // p.y is the monotonically descending bed target, so the
                    // surface is continuous by construction.
                    float y = Mathf.Max(p.y + settings.RiverDepth * 0.55f, settings.SeaLevel + 0.02f);

                    float hw = p.w * 0.8f;
                    verts.Add(new Vector3(p.x + side.x * hw, y, p.z + side.y * hw));
                    verts.Add(new Vector3(p.x - side.x * hw, y, p.z - side.y * hw));

                    if (i > 0)
                    {
                        int b = verts.Count - 4;
                        tris.AddRange(new[] { b, b + 2, b + 1, b + 1, b + 2, b + 3 });
                    }
                }

                var mesh = new Mesh { name = "RiverWater" };
                mesh.SetVertices(verts);
                mesh.SetTriangles(tris, 0);
                mesh.RecalculateNormals();
                mesh.RecalculateBounds();

                var go = new GameObject("RiverWater") { hideFlags = HideFlags.DontSave };
                go.transform.SetParent(transform, false);
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
                go.AddComponent<MeshRenderer>().sharedMaterial = waterMaterial;
            }
        }

        public Color SampleBiomeColor(float x, float z, float height)
        {
            float h = height - settings.SeaLevel;
            float moisture = SampleMoisture(x, z);
            float river = RiverInfluence01(x, z);

            // River channels read as wet mud banks (blended, not a hard cut).
            if (river > 0.25f && h > 0f)
            {
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
