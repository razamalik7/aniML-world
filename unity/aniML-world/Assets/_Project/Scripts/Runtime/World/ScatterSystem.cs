using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace AniML.World
{
    /// <summary>
    /// Deterministic, instanced prop scatter driven by the world-truth
    /// functions. No GameObjects per prop: instances are matrices rendered
    /// with GPU instancing per chunk (frustum-culled), plus a lightweight
    /// record list the simulation can query and (later) mutate — a tree the
    /// sim removes stops rendering, because the sim data IS the render data.
    /// </summary>
    [RequireComponent(typeof(ProceduralWorld))]
    public sealed class ScatterSystem : MonoBehaviour
    {
        [SerializeField] private ScatterConfig config;

        public struct Instance
        {
            public int LayerIndex;
            public Vector3 Position;
            public float Scale;
            public bool Removed;
        }

        private sealed class RenderBatch
        {
            public Mesh Mesh;
            public int SubMesh;
            public Material Material;
            public Matrix4x4 LocalPart;          // renderer offset inside prefab
            public readonly List<Matrix4x4> Matrices = new();
            // Parallel to Matrices: which layer each transform came from,
            // so fire (and later the ecosystem) can remove vegetation only.
            public readonly List<int> MatrixLayers = new();
        }

        private sealed class Chunk
        {
            public Bounds Bounds;
            public readonly List<Instance> Instances = new();
            public readonly List<RenderBatch> Batches = new();
        }

        private sealed class VariantRenderer
        {
            public Mesh Mesh;
            public int SubMesh;
            public Material Material;
            public Matrix4x4 Local;
        }

        private ProceduralWorld _world;
        private readonly List<Chunk> _chunks = new();
        private readonly List<VariantRenderer>[][] _variantCache = new List<VariantRenderer>[64][];
        private readonly Plane[] _frustum = new Plane[6];
        private bool _generated;

        public ScatterConfig Config => config;

        public void Configure(ScatterConfig c) => config = c;

        private void OnEnable()
        {
            if (!_generated && Application.isPlaying)
                Generate();
        }

        private void LateUpdate()
        {
            // Submit this frame's instanced draws (all cameras).
            SubmitDraws(null);
        }

        [ContextMenu("Generate")]
        public void Generate()
        {
            _world = GetComponent<ProceduralWorld>();
            var settings = _world.Settings;
            if (config == null || settings == null)
            {
                Debug.LogError("[AniML] ScatterSystem missing config or world settings.");
                return;
            }

            // World functions must be initialized (noise fields live there).
            _world.EnsureSamplers();

            _chunks.Clear();
            int chunksPerSide = Mathf.RoundToInt(settings.WorldSize / settings.ChunkSize);
            float half = settings.WorldSize * 0.5f;

            int totalInstances = 0;
            for (int cz = 0; cz < chunksPerSide; cz++)
            {
                for (int cx = 0; cx < chunksPerSide; cx++)
                {
                    var origin = new Vector2(cx * settings.ChunkSize - half, cz * settings.ChunkSize - half);
                    var chunk = BuildChunk(cx, cz, origin, settings);
                    if (chunk != null)
                    {
                        _chunks.Add(chunk);
                        totalInstances += chunk.Instances.Count;
                    }
                }
            }

            EmitCoastCliffs();
            EmitInlandCliffs();

            _generated = true;
            Debug.Log($"[AniML] Scatter generated: {totalInstances} instances across {_chunks.Count} chunks "
                      + $"(+{_world.CoastCliffs.Count} coastal, +{_world.InlandCliffs.Count} inland cliff segments).");

            // Per-layer census: proves a layer actually spawns (a typo'd
            // prefab name or an impossible climate gate shows up as 0).
            var perLayer = new int[config.Layers.Count];
            foreach (var chunk in _chunks)
                foreach (var c in chunk.Instances)
                    perLayer[c.LayerIndex]++;
            var census = new System.Text.StringBuilder("[AniML] Scatter census:");
            for (int li = 0; li < perLayer.Length; li++)
                if (!config.Layers[li].CoastalWall && !config.Layers[li].InlandWall)
                    census.Append($" {config.Layers[li].Name}={perLayer[li]}");
            Debug.Log(census.ToString());
        }

        private struct Candidate
        {
            public int LayerIndex;
            public int Variant;
            public Vector3 Pos;
            public float Scale;
            public float Yaw;
            public float TiltX;
            public float TiltZ;
            public float Canopy;   // competition radius; 0 = exempt
        }

        private Chunk BuildChunk(int cx, int cz, Vector2 origin, WorldSettings settings)
        {
            var candidates = new List<Candidate>();

            for (int li = 0; li < config.Layers.Count; li++)
            {
                var layer = config.Layers[li];
                if (!layer.Enabled || layer.Prefabs.Count == 0 || layer.DensityPerSqm <= 0f || layer.CoastalWall || layer.InlandWall)
                    continue;

                var rng = new System.Random(HashCell(settings.Seed, cx, cz, li));
                float area = settings.ChunkSize * settings.ChunkSize;
                float expected = layer.DensityPerSqm * area;
                int samples = Mathf.CeilToInt(expected);
                float acceptBase = expected / samples;

                for (int i = 0; i < samples; i++)
                {
                    float x = origin.x + (float)rng.NextDouble() * settings.ChunkSize;
                    float z = origin.y + (float)rng.NextDouble() * settings.ChunkSize;

                    float h = _world.SampleHeight(x, z);
                    float hAboveSea = h - settings.SeaLevel;
                    if (hAboveSea < layer.MinHeight || hAboveSea > layer.MaxHeight)
                        continue;

                    if (SlopeDeg(x, z) > layer.MaxSlopeDeg)
                        continue;

                    // Nothing grows in the river channel itself.
                    if (_world.RiverInfluence01(x, z) > 0.3f)
                        continue;

                    // ...or drowned under a lake/pond. Waders (reeds) tolerate
                    // shallows up to their MaxWaterDepth; floaters (lily
                    // pads) instead REQUIRE water and ride its surface.
                    float wDepth = _world.StandingWaterDepth(x, z);
                    if (layer.OnWaterSurface)
                    {
                        if (wDepth < layer.MinWaterDepth)
                            continue;
                    }
                    else if (wDepth > layer.MaxWaterDepth)
                        continue;

                    // Species grow only in their biomes.
                    if (layer.AllowedBiomes.Count > 0 && !layer.AllowedBiomes.Contains(_world.SampleBiome(x, z)))
                        continue;

                    // ...and only in their temperature band (soft edges).
                    if (layer.MinTemp01 > 0f || layer.MaxTemp01 < 1f)
                    {
                        float t = _world.SampleTemperature01(x, z);
                        float band = Mathf.Min(
                            Mathf.InverseLerp(layer.MinTemp01 - 0.06f, layer.MinTemp01 + 0.06f, t),
                            Mathf.InverseLerp(layer.MaxTemp01 + 0.06f, layer.MaxTemp01 - 0.06f, t));
                        if (band <= 0f) continue;
                        if (rng.NextDouble() > band) continue;
                    }

                    float accept = acceptBase;
                    if (layer.DriveByForestDensity)
                    {
                        // Forests are objects: density comes from the forest
                        // field (0 on open plains, sparse parkland at edges).
                        accept *= Mathf.Pow(_world.ForestDensity01(x, z), 1.1f);
                    }
                    else if (layer.MoistureFull > layer.MoistureStart)
                    {
                        float m = _world.SampleMoisture(x, z);
                        accept *= Mathf.Clamp01(Mathf.InverseLerp(layer.MoistureStart, layer.MoistureFull, m));
                    }

                    // Water need: density peaks at the species' preferred
                    // moisture and falls away from it — gradients WITHIN
                    // biomes, not uniform fill.
                    if (layer.WaterNeed >= 0f)
                    {
                        float m = _world.SampleMoisture(x, z);
                        accept *= Mathf.Clamp01(1f - Mathf.Abs(m - layer.WaterNeed) * 2.2f);
                    }

                    if (layer.ForestSuppression > 0f)
                        accept *= 1f - layer.ForestSuppression * _world.ForestDensity01(x, z);

                    // Little grows in volcanic ash.
                    if (layer.IsVegetation)
                        accept *= 1f - _world.VolcanoAsh01(x, z) * 0.92f;

                    if (layer.PreferRocky)
                        accept *= 0.12f + 0.88f * _world.SampleCragginess01(x, z);

                    if (rng.NextDouble() > accept)
                        continue;

                    float scale = Mathf.Lerp(layer.ScaleRange.x, layer.ScaleRange.y, (float)rng.NextDouble());
                    float baseY = layer.OnWaterSurface ? h + wDepth + 0.05f : h;
                    candidates.Add(new Candidate
                    {
                        LayerIndex = li,
                        Variant = rng.Next(layer.Prefabs.Count),
                        Pos = new Vector3(x, baseY + layer.YOffset * scale, z),
                        Scale = scale,
                        Yaw = (float)rng.NextDouble() * 360f,
                        TiltX = ((float)rng.NextDouble() - 0.5f) * 2f * layer.MaxTiltDeg,
                        TiltZ = ((float)rng.NextDouble() - 0.5f) * 2f * layer.MaxTiltDeg,
                        Canopy = layer.CanopyRadius * scale,
                    });
                }
            }

            if (candidates.Count == 0)
                return null;

            // Competition self-thinning (Deussen-style, cheap version):
            // larger canopies win overlaps — big trees space out, saplings
            // and shrubs survive only in the gaps. Grass (canopy 0) exempt.
            var kept = new List<Candidate>(candidates.Count);
            var competitors = new List<Candidate>();
            foreach (var c in candidates)
            {
                if (c.Canopy <= 0f) kept.Add(c);
                else competitors.Add(c);
            }
            competitors.Sort((a, b) => b.Canopy.CompareTo(a.Canopy));
            var winners = new List<Candidate>();
            foreach (var c in competitors)
            {
                bool shaded = false;
                foreach (var w in winners)
                {
                    float dx = c.Pos.x - w.Pos.x, dz = c.Pos.z - w.Pos.z;
                    float minDist = (c.Canopy + w.Canopy) * 0.62f;
                    if (dx * dx + dz * dz < minDist * minDist) { shaded = true; break; }
                }
                if (!shaded) winners.Add(c);
            }
            kept.AddRange(winners);

            // Emit survivors into render batches + sim records.
            var chunk = new Chunk();
            float minY = float.MaxValue, maxY = float.MinValue;
            foreach (var c in kept)
            {
                var layer = config.Layers[c.LayerIndex];
                // Deterministic per-position Y-stretch: same mesh, different build.
                float stretch = 1f + layer.YStretch *
                    (Mathf.PerlinNoise(c.Pos.x * 0.173f + 71f, c.Pos.z * 0.173f - 37f) * 2f - 1f);
                var scale = new Vector3(c.Scale, c.Scale * stretch, c.Scale);
                var matrix = Matrix4x4.TRS(c.Pos, Quaternion.Euler(c.TiltX, c.Yaw, c.TiltZ), scale);
                chunk.Instances.Add(new Instance { LayerIndex = c.LayerIndex, Position = c.Pos, Scale = c.Scale });

                foreach (var vr in GetVariantRenderers(c.LayerIndex, c.Variant, layer))
                {
                    var batch = FindBatch(chunk, vr);
                    batch.Matrices.Add(matrix * vr.Local);
                    batch.MatrixLayers.Add(c.LayerIndex);
                }

                minY = Mathf.Min(minY, c.Pos.y);
                maxY = Mathf.Max(maxY, c.Pos.y + 20f * c.Scale);
            }

            var center = new Vector3(origin.x + settings.ChunkSize * 0.5f, (minY + maxY) * 0.5f, origin.y + settings.ChunkSize * 0.5f);
            var size = new Vector3(settings.ChunkSize + 20f, Mathf.Max(maxY - minY, 10f) + 20f, settings.ChunkSize + 20f);
            chunk.Bounds = new Bounds(center, size);
            return chunk;
        }

        /// <summary>Coastal cliff walls: instances come from the world's coast
        /// tracer (aligned, chained) instead of random sampling.</summary>
        private void EmitCoastCliffs()
        {
            int li = config.Layers.FindIndex(l => l.CoastalWall && l.Prefabs.Count > 0);
            if (li < 0 || _world.CoastCliffs.Count == 0)
                return;

            var layer = config.Layers[li];
            var chunk = new Chunk();
            var bounds = new Bounds(Vector3.zero, Vector3.zero);
            var rng = new System.Random(_world.Settings.Seed * 61 + 29);

            foreach (var cliff in _world.CoastCliffs)
            {
                float scale = cliff.Scale * Mathf.Lerp(layer.ScaleRange.x, layer.ScaleRange.y, (float)rng.NextDouble());
                var pos = cliff.Pos + Vector3.up * layer.YOffset * scale;
                var matrix = Matrix4x4.TRS(pos, Quaternion.Euler(0f, cliff.YawDeg, 0f), Vector3.one * scale);
                chunk.Instances.Add(new Instance { LayerIndex = li, Position = pos, Scale = scale });

                foreach (var vr in GetVariantRenderers(li, rng.Next(layer.Prefabs.Count), layer))
                {
                    var batch = FindBatch(chunk, vr);
                    batch.Matrices.Add(matrix * vr.Local);
                    batch.MatrixLayers.Add(li);
                }
                bounds.Encapsulate(new Bounds(pos, Vector3.one * 40f * scale));
            }

            chunk.Bounds = bounds;
            _chunks.Add(chunk);
        }

        /// <summary>Inland cliff walls: chained, aligned instances from the
        /// world's slope tracer — one instanced layer per rock family.</summary>
        private void EmitInlandCliffs()
        {
            if (_world.InlandCliffs.Count == 0)
                return;

            var rng = new System.Random(_world.Settings.Seed * 67 + 31);
            for (int li = 0; li < config.Layers.Count; li++)
            {
                var layer = config.Layers[li];
                if (!layer.InlandWall || layer.Prefabs.Count == 0)
                    continue;

                var chunk = new Chunk();
                var bounds = new Bounds(Vector3.zero, Vector3.zero);
                int count = 0;
                foreach (var cliff in _world.InlandCliffs)
                {
                    if (cliff.Family != layer.InlandFamily)
                        continue;
                    float scale = cliff.Scale * Mathf.Lerp(layer.ScaleRange.x, layer.ScaleRange.y, (float)rng.NextDouble());
                    var pos = cliff.Pos + Vector3.up * layer.YOffset * scale;
                    // Lean tops 9° into the hill so downhill edges stay
                    // seated without the crowns overhanging.
                    var rot = Quaternion.Euler(0f, cliff.YawDeg, 0f) * Quaternion.Euler(-9f, 0f, 0f);
                    var matrix = Matrix4x4.TRS(pos, rot, Vector3.one * scale);
                    chunk.Instances.Add(new Instance { LayerIndex = li, Position = pos, Scale = scale });

                    foreach (var vr in GetVariantRenderers(li, rng.Next(layer.Prefabs.Count), layer))
                    {
                        var batch = FindBatch(chunk, vr);
                        batch.Matrices.Add(matrix * vr.Local);
                        batch.MatrixLayers.Add(li);
                    }
                    bounds.Encapsulate(new Bounds(pos, Vector3.one * 40f * scale));
                    count++;
                }

                if (count > 0)
                {
                    chunk.Bounds = bounds;
                    _chunks.Add(chunk);
                }
            }
        }

        /// <summary>
        /// Remove vegetation instances within a radius (fire, and later
        /// grazing/trampling). Rocks, walls, and props survive.
        /// </summary>
        public void RemoveVegetationInRadius(Vector3 center, float radius)
        {
            float r2 = radius * radius;
            foreach (var chunk in _chunks)
            {
                if ((chunk.Bounds.ClosestPoint(center) - center).sqrMagnitude > r2)
                    continue;
                foreach (var batch in chunk.Batches)
                {
                    for (int i = batch.Matrices.Count - 1; i >= 0; i--)
                    {
                        int li = batch.MatrixLayers[i];
                        if (li < 0 || li >= config.Layers.Count || !config.Layers[li].IsVegetation)
                            continue;
                        var m = batch.Matrices[i];
                        float dx = m.m03 - center.x, dz = m.m23 - center.z;
                        if (dx * dx + dz * dz > r2)
                            continue;
                        batch.Matrices.RemoveAt(i);
                        batch.MatrixLayers.RemoveAt(i);
                    }
                }
                chunk.Instances.RemoveAll(inst =>
                {
                    if (inst.LayerIndex < 0 || inst.LayerIndex >= config.Layers.Count
                        || !config.Layers[inst.LayerIndex].IsVegetation)
                        return false;
                    float dx = inst.Position.x - center.x, dz = inst.Position.z - center.z;
                    return dx * dx + dz * dz <= r2;
                });
            }
        }

        // ---- Sim-facing API (grows with the ecosystem) ----

        public void ForEachInstanceInRadius(Vector3 center, float radius, System.Action<Instance> visit)
        {
            float r2 = radius * radius;
            foreach (var chunk in _chunks)
            {
                if (chunk.Bounds.SqrDistance(center) > r2)
                    continue;
                foreach (var inst in chunk.Instances)
                {
                    if (!inst.Removed && (inst.Position - center).sqrMagnitude <= r2)
                        visit(inst);
                }
            }
        }

        // ---- Rendering ----

        /// <summary>
        /// Enqueue instanced draws for the current frame. Pass a camera to
        /// restrict draws to it (used by the headless capture pipeline);
        /// null draws into every camera with culling against Camera.main.
        /// </summary>
        public void SubmitDraws(Camera cam)
        {
            if (!_generated)
                return;

            var cullCam = cam != null ? cam : Camera.main;
            bool hasFrustum = cullCam != null;
            if (hasFrustum)
                GeometryUtility.CalculateFrustumPlanes(cullCam, _frustum);

            foreach (var chunk in _chunks)
            {
                if (hasFrustum && !GeometryUtility.TestPlanesAABB(_frustum, chunk.Bounds))
                    continue;

                foreach (var batch in chunk.Batches)
                {
                    var rp = new RenderParams(batch.Material)
                    {
                        camera = cam,
                        worldBounds = chunk.Bounds,
                        shadowCastingMode = ShadowCastingMode.On,
                        receiveShadows = true,
                    };

                    var list = batch.Matrices;
                    for (int start = 0; start < list.Count; start += 1000)
                    {
                        int count = Mathf.Min(1000, list.Count - start);
                        Graphics.RenderMeshInstanced(rp, batch.Mesh, batch.SubMesh, list, count, start);
                    }
                }
            }
        }

        // ---- Prefab variant extraction (LOD0 meshes + materials) ----

        private IEnumerable<VariantRenderer> GetVariantRenderers(int layerIndex, int variant, ScatterLayer layer)
        {
            _variantCache[layerIndex] ??= new List<VariantRenderer>[Mathf.Max(layer.Prefabs.Count, 1)];
            var cache = _variantCache[layerIndex];
            if (cache[variant] != null)
                return cache[variant];

            var result = new List<VariantRenderer>();
            var prefab = layer.Prefabs[variant];
            if (prefab != null)
            {
                Renderer[] renderers;
                var lodGroup = prefab.GetComponentInChildren<LODGroup>();
                if (lodGroup != null && lodGroup.GetLODs().Length > 0)
                    renderers = lodGroup.GetLODs()[0].renderers;
                else
                    renderers = prefab.GetComponentsInChildren<MeshRenderer>();

                var rootInv = prefab.transform.worldToLocalMatrix;
                foreach (var r in renderers)
                {
                    if (r == null) continue;
                    var mf = r.GetComponent<MeshFilter>();
                    if (mf == null || mf.sharedMesh == null) continue;

                    var local = rootInv * r.transform.localToWorldMatrix;
                    var mats = r.sharedMaterials;
                    int subCount = Mathf.Min(mf.sharedMesh.subMeshCount, mats.Length);
                    for (int s = 0; s < subCount; s++)
                    {
                        if (mats[s] == null) continue;
                        var mat = layer.OverrideMaterial != null ? layer.OverrideMaterial : mats[s];
                        mat.enableInstancing = true;
                        result.Add(new VariantRenderer
                        {
                            Mesh = mf.sharedMesh,
                            SubMesh = s,
                            Material = mat,
                            Local = local,
                        });
                    }
                }
            }

            cache[variant] = result;
            return result;
        }

        private static RenderBatch FindBatch(Chunk chunk, VariantRenderer vr)
        {
            foreach (var b in chunk.Batches)
            {
                if (b.Mesh == vr.Mesh && b.SubMesh == vr.SubMesh && b.Material == vr.Material && b.LocalPart == vr.Local)
                    return b;
            }
            var batch = new RenderBatch { Mesh = vr.Mesh, SubMesh = vr.SubMesh, Material = vr.Material, LocalPart = vr.Local };
            chunk.Batches.Add(batch);
            return batch;
        }

        private float SlopeDeg(float x, float z)
        {
            const float e = 1.5f;
            float hx = _world.SampleHeight(x + e, z) - _world.SampleHeight(x - e, z);
            float hz = _world.SampleHeight(x, z + e) - _world.SampleHeight(x, z - e);
            var normal = new Vector3(-hx, 2f * e, -hz).normalized;
            return Vector3.Angle(normal, Vector3.up);
        }

        private static int HashCell(int seed, int cx, int cz, int layer)
        {
            unchecked
            {
                int h = seed;
                h = h * 486187739 + cx;
                h = h * 486187739 + cz;
                h = h * 486187739 + layer;
                return h;
            }
        }
    }
}
