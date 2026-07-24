using System.IO;
using AniML.World;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AniML.EditorTools
{
    /// <summary>
    /// Editor pipeline for the procedural world: ensure scene objects,
    /// generate terrain, and capture verification renders (works headless
    /// in batchmode as long as -nographics is NOT used).
    /// </summary>
    public static class WorldTools
    {
        private const string ScenePath = "Assets/_Project/Scenes/Sandbox.unity";
        private const string SettingsPath = "Assets/_Project/Settings/WorldSettings.asset";
        private const string MaterialPath = "Assets/_Project/Art/Materials/M_Terrain.mat";
        private const string WaterMaterialPath = "Assets/_Project/Art/Materials/M_Water_Placeholder.mat";

        [MenuItem("AniML/World/Generate In Sandbox Scene")]
        public static void GenerateInScene()
        {
            var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

            var settings = LoadOrCreateSettings();
            var material = LoadOrCreateTerrainMaterial();

            // Legacy flat ground from the first bootstrap is superseded.
            var oldGround = GameObject.Find("Ground");
            if (oldGround != null)
                Object.DestroyImmediate(oldGround);

            var root = GameObject.Find("World");
            if (root == null)
                root = new GameObject("World");

            var world = root.GetComponent<ProceduralWorld>();
            if (world == null)
                world = root.AddComponent<ProceduralWorld>();

            world.Configure(settings, material, EnsureWaterMaterial());
            world.Generate();

            var scatter = root.GetComponent<ScatterSystem>();
            if (scatter == null)
                scatter = root.AddComponent<ScatterSystem>();
            scatter.Configure(EnsureScatterConfig());
            scatter.Generate();

            var wonders = root.GetComponent<WonderPlacer>();
            if (wonders == null)
                wonders = root.AddComponent<WonderPlacer>();
            WireWonderPrefabs(wonders);
            wonders.Place(world);

            EnableFoliageColourVariation();
            EnsureWaterPlane(settings);

            // Warm trilight ambient lifts saturation vs the default grey SH.
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.62f, 0.72f, 0.85f);
            RenderSettings.ambientEquatorColor = new Color(0.55f, 0.58f, 0.52f);
            RenderSettings.ambientGroundColor = new Color(0.32f, 0.30f, 0.26f);

            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
            Debug.Log("[AniML] World generated and scene saved.");
        }

        /// <summary>Batchmode entry: generate + save + render verification shots.</summary>
        public static void GenerateAndCapture()
        {
            GenerateInScene();

            string outDir = System.Environment.GetEnvironmentVariable("ANIML_SHOT_DIR");
            if (string.IsNullOrEmpty(outDir))
                outDir = Path.Combine(Path.GetTempPath(), "animl_world_shots");
            Directory.CreateDirectory(outDir);

            Capture(new Vector3(0f, 620f, -1f), Quaternion.Euler(90f, 0f, 0f), 60f,
                Path.Combine(outDir, "world_top.png"), orthographic: true, orthoSize: 540f);
            Capture(new Vector3(-380f, 95f, -420f), Quaternion.Euler(14f, 38f, 0f), 55f,
                Path.Combine(outDir, "world_perspective.png"));
            Capture(new Vector3(30f, 32f, -180f), Quaternion.Euler(9f, -12f, 0f), 60f,
                Path.Combine(outDir, "world_ground.png"));

            // Close-ups of every wonder + the primary summit, framed for review.
            var worldComp = Object.FindFirstObjectByType<ProceduralWorld>();
            if (worldComp != null)
            {
                foreach (var site in worldComp.Wonders)
                {
                    Vector3 eye;
                    if (site.Kind == "Waterfall")
                    {
                        // Stand downstream, look back up at the falls.
                        float rad = site.YawDeg * Mathf.Deg2Rad;
                        var downstream = new Vector3(Mathf.Sin(rad), 0f, Mathf.Cos(rad));
                        eye = site.Pos + downstream * 42f + Vector3.up * 12f;
                    }
                    else
                    {
                        eye = site.Pos + new Vector3(34f, 26f, -34f);
                    }
                    Capture(eye, Quaternion.LookRotation(site.Pos + Vector3.up * 5f - eye), 50f,
                        Path.Combine(outDir, $"wonder_{site.Kind}.png"));
                }
                if (worldComp.Summits.Count > 0)
                {
                    var s = worldComp.Summits[0];
                    var eye = s.Pos + new Vector3(-160f, 40f, -160f);
                    Capture(eye, Quaternion.LookRotation(s.Pos + Vector3.up * 20f - eye), 55f,
                        Path.Combine(outDir, "summit.png"));
                }
            }

            ExportBiomeMap();
            DumpLandformMeshBounds();
            Debug.Log($"[AniML] Wrote verification renders to {outDir}");
        }

        /// <summary>
        /// Log the AABBs of the landform-candidate meshes (mountain ranges,
        /// cliff walls, cave, arches) so placement scales can be calibrated
        /// without opening the editor.
        /// </summary>
        [MenuItem("AniML/World/Dump Landform Mesh Bounds")]
        public static void DumpLandformMeshBounds()
        {
            void Dump(string label, string root, params string[] names)
            {
                foreach (var prefab in FindPrefabs(root, names))
                {
                    var b = new Bounds();
                    bool first = true;
                    foreach (var r in prefab.GetComponentsInChildren<MeshRenderer>())
                    {
                        if (first) { b = r.bounds; first = false; }
                        else b.Encapsulate(r.bounds);
                    }
                    Debug.Log($"[AniML] BOUNDS {label}/{prefab.name}: size={b.size:F1} center={b.center:F1}");
                }
            }

            Dump("Alpine", AlpineRoot, "SM_Env_MountainRange_01", "SM_Env_MountainRange_02",
                "SM_Env_MountainRange_03", "SM_Env_MountainRange_04",
                "SM_Env_Rock_Rough_01", "SM_Env_Rock_Rough_02", "SM_Env_Rock_01",
                "SM_Env_Rock_02", "SM_Env_Rock_05", "SM_Env_Rock_010",
                "SM_Env_Rock_Cliff_Arch_01");
            Dump("Arid", AridRoot, "SM_Env_Rock_Cliff_01", "SM_Env_Rock_Cliff_05",
                "SM_Env_Rock_Cliff_10", "SM_Env_Rock_Cliff_16", "SM_Env_Rock_Arch_01");
            Dump("Tropical", TropicalRoot, "SM_Env_Rock_Cave_01", "SM_Env_Rock_Cliff_01");
            Dump("Meadow", MeadowRoot, "SM_Env_Rock_Cliff_01", "SM_Env_Rock_Cliff_02", "SM_Env_Rock_Cliff_03");
        }

        private static void Capture(Vector3 pos, Quaternion rot, float fov, string path,
            bool orthographic = false, float orthoSize = 100f)
        {
            var go = new GameObject("CaptureCamera");
            var cam = go.AddComponent<Camera>();
            go.transform.SetPositionAndRotation(pos, rot);
            cam.fieldOfView = fov;
            cam.orthographic = orthographic;
            cam.orthographicSize = orthoSize;
            cam.nearClipPlane = 0.3f;
            cam.farClipPlane = 4000f;
            cam.clearFlags = CameraClearFlags.Skybox;

            const int w = 1600, h = 900;
            var rt = new RenderTexture(w, h, 24);
            cam.targetTexture = rt;

            // Instanced scatter draws are frame-scoped: enqueue for this
            // camera right before rendering it.
            var scatter = Object.FindFirstObjectByType<ScatterSystem>();
            if (scatter != null)
                scatter.SubmitDraws(cam);

            cam.Render();

            RenderTexture.active = rt;
            var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
            tex.Apply();
            RenderTexture.active = null;

            File.WriteAllBytes(path, tex.EncodeToPNG());

            cam.targetTexture = null;
            Object.DestroyImmediate(rt);
            Object.DestroyImmediate(tex);
            Object.DestroyImmediate(go);
        }

        private const string ScatterConfigPath = "Assets/_Project/Settings/ScatterConfig.asset";

        /// <summary>One inland cliff-wall layer (instances come from the
        /// slope tracer, so only family, prefabs, and looks matter).</summary>
        private static ScatterLayer CliffWallLayer(string name, CliffFamily family,
            string root, Vector2 scaleRange, params string[] prefabNames)
        {
            return new ScatterLayer
            {
                Name = name,
                Prefabs = FindPrefabs(root, prefabNames),
                InlandWall = true,
                InlandFamily = family,
                YOffset = -0.1f,
                ScaleRange = scaleRange,
                IsVegetation = false,
            };
        }

        private static ScatterLayer[] BuildCliffWallLayers() => new[]
        {
            // NOTE: Alpine's "Rock_Cliff" set is GLACIER ICE (refractive
            // shader) — real granite is the plain Rock / Rock_Rough meshes,
            // scaled up to wall size.
            CliffWallLayer("CliffWallGranite", CliffFamily.Granite, AlpineRoot,
                new Vector2(1.5f, 2.2f),
                "SM_Env_Rock_Rough_01", "SM_Env_Rock_Rough_02",
                "SM_Env_Rock_01", "SM_Env_Rock_02", "SM_Env_Rock_05", "SM_Env_Rock_010"),
            CliffWallLayer("CliffWallRedRock", CliffFamily.RedRock, AridRoot,
                new Vector2(0.85f, 1.2f),
                "SM_Env_Rock_Cliff_01", "SM_Env_Rock_Cliff_02", "SM_Env_Rock_Cliff_03",
                "SM_Env_Rock_Cliff_04", "SM_Env_Rock_Cliff_05", "SM_Env_Rock_Cliff_06",
                "SM_Env_Rock_Cliff_07", "SM_Env_Rock_Cliff_08", "SM_Env_Rock_Cliff_09",
                "SM_Env_Rock_Cliff_10", "SM_Env_Rock_Cliff_11", "SM_Env_Rock_Cliff_12",
                "SM_Env_Rock_Cliff_13", "SM_Env_Rock_Cliff_14", "SM_Env_Rock_Cliff_15",
                "SM_Env_Rock_Cliff_16"),
            CliffWallLayer("CliffWallMossy", CliffFamily.Mossy, MeadowRoot,
                new Vector2(0.85f, 1.2f),
                "SM_Env_Rock_Cliff_01", "SM_Env_Rock_Cliff_02", "SM_Env_Rock_Cliff_03"),
        };

        /// <summary>
        /// Idempotent migration for pre-existing config assets: wetland
        /// waders stand in the shallows (serialized MaxWaterDepth defaults
        /// to 0 = dry ground only), and the inland cliff-wall layers are
        /// appended if the asset predates the slope tracer.
        /// </summary>
        private static void MigrateConfig(ScatterConfig cfg)
        {
            var depths = new System.Collections.Generic.Dictionary<string, float>
            {
                ["WetlandReeds"] = 0.5f,
                ["WetlandMoss"] = 0.15f,
                ["WetlandTrees"] = 0.35f,
            };
            bool dirty = false;
            foreach (var layer in cfg.Layers)
            {
                if (depths.TryGetValue(layer.Name, out float d) && layer.MaxWaterDepth != d)
                {
                    layer.MaxWaterDepth = d;
                    dirty = true;
                }
            }
            // The builder is the source of truth for wall layers: replace in
            // place so prefab-set corrections propagate to existing assets.
            foreach (var wall in BuildCliffWallLayers())
            {
                int idx = cfg.Layers.FindIndex(l => l.Name == wall.Name);
                if (idx >= 0)
                    cfg.Layers[idx] = wall;
                else
                    cfg.Layers.Add(wall);
                dirty = true;
            }
            if (dirty)
            {
                EditorUtility.SetDirty(cfg);
                AssetDatabase.SaveAssets();
            }
        }

        private static ScatterConfig EnsureScatterConfig()
        {
            var cfg = AssetDatabase.LoadAssetAtPath<ScatterConfig>(ScatterConfigPath);
            if (cfg != null)
            {
                MigrateConfig(cfg);
                return cfg;
            }

            cfg = ScriptableObject.CreateInstance<ScatterConfig>();
            var L = cfg.Layers;

            // ---- Forest / meadow flora (Meadow pack) ----
            // Southern mixed forests: broadleaf-dominated, fruit accents.
            L.Add(new ScatterLayer
            {
                Name = "ForestTreesMixed",
                Prefabs = FindPrefabs(MeadowRoot, "SM_Env_Tree_Meadow_01", "SM_Env_Tree_Meadow_02",
                    "SM_Env_Tree_Fruit_01", "SM_Env_Tree_Birch_01"),
                MinHeight = 3f, MaxHeight = 38f, MaxSlopeDeg = 27f,
                DriveByForestDensity = true, DensityPerSqm = 0.05f,
                CanopyRadius = 3.2f, MinTemp01 = 0.42f, YStretch = 0.16f,
                ScaleRange = new Vector2(0.85f, 1.45f),
            });
            // Northern birch forests: pale trunks, colder identity.
            L.Add(new ScatterLayer
            {
                Name = "ForestTreesBirch",
                Prefabs = FindPrefabs(MeadowRoot, "SM_Env_Tree_Birch_01", "SM_Env_Tree_Birch_02",
                    "SM_Env_Tree_Birch_03", "SM_Env_Tree_Meadow_01"),
                MinHeight = 3f, MaxHeight = 38f, MaxSlopeDeg = 27f,
                DriveByForestDensity = true, DensityPerSqm = 0.05f,
                CanopyRadius = 3f, MaxTemp01 = 0.5f, YStretch = 0.18f,
                ScaleRange = new Vector2(0.8f, 1.4f),
            });
            // Lush canopy accents where forests are wettest.
            L.Add(new ScatterLayer
            {
                Name = "ForestTreesLush",
                Prefabs = FindPrefabs(TropicalRoot, "SM_Env_Tree_Forest_01", "SM_Env_Tree_Forest_02",
                    "SM_Env_Tree_Forest_03"),
                MinHeight = 3f, MaxHeight = 30f, MaxSlopeDeg = 24f,
                DriveByForestDensity = true, DensityPerSqm = 0.012f,
                WaterNeed = 0.78f, CanopyRadius = 3.6f,
                MinTemp01 = 0.5f, YStretch = 0.14f,
                ScaleRange = new Vector2(0.9f, 1.5f),
            });
            // Palms on the warm southern shores only.
            L.Add(new ScatterLayer
            {
                Name = "BeachPalms",
                Prefabs = FindPrefabs(TropicalRoot, "SM_Env_Tree_Palm_01", "SM_Env_Tree_Palm_02",
                    "SM_Env_Tree_Palm_03"),
                MinHeight = 0.8f, MaxHeight = 6f, MaxSlopeDeg = 22f,
                AllowedBiomes = { Biome.Beach }, MinTemp01 = 0.62f,
                DensityPerSqm = 0.002f, CanopyRadius = 2.6f, YStretch = 0.2f, MaxTiltDeg = 4f,
                ScaleRange = new Vector2(0.85f, 1.3f),
            });
            L.Add(new ScatterLayer
            {
                Name = "FruitTrees",
                Prefabs = FindPrefabs(MeadowRoot, "SM_Env_Tree_Fruit_01", "SM_Env_Tree_Fruit_02", "SM_Env_Tree_Fruit_03"),
                MinHeight = 3f, MaxHeight = 24f, MaxSlopeDeg = 20f,
                AllowedBiomes = { Biome.Meadow }, WaterNeed = 0.55f,
                DensityPerSqm = 0.0012f, CanopyRadius = 2.6f, YStretch = 0.14f,
                ScaleRange = new Vector2(0.9f, 1.2f),
            });
            L.Add(new ScatterLayer
            {
                Name = "Bushes",
                Prefabs = FindPrefabs(MeadowRoot, "SM_Env_Bush_01", "SM_Env_Bush_02", "SM_Env_Bush_03", "SM_Env_Grass_Bush_01"),
                MinHeight = 3f, MaxHeight = 32f, MaxSlopeDeg = 32f,
                AllowedBiomes = { Biome.Meadow, Biome.ForestFloor, Biome.Wetland },
                WaterNeed = 0.5f, DensityPerSqm = 0.02f, CanopyRadius = 1.3f,
                ScaleRange = new Vector2(0.8f, 1.4f),
            });
            L.Add(new ScatterLayer
            {
                Name = "GrassClumps",
                Prefabs = FindPrefabs(MeadowRoot,
                    "SM_Env_Grass_Tall_Clump_01", "SM_Env_Grass_Tall_Clump_02", "SM_Env_Grass_Tall_Clump_03",
                    "SM_Env_Grass_Med_Clump_01", "SM_Env_Grass_Med_Clump_02", "SM_Env_Grass_Med_Clump_03",
                    "SM_Env_Grass_Short_Clump_01", "SM_Env_Grass_Short_Clump_02", "SM_Env_Grass_Short_Clump_03",
                    "SM_Env_Grass_Tall_Clump_04", "SM_Env_Grass_Tall_Clump_05"),
                MinHeight = 2.5f, MaxHeight = 35f, MaxSlopeDeg = 33f,
                AllowedBiomes = { Biome.Meadow, Biome.ForestFloor, Biome.AlpineMeadow, Biome.Beach },
                WaterNeed = 0.45f, DensityPerSqm = 0.22f, ForestSuppression = 0.55f,
                ScaleRange = new Vector2(0.9f, 1.7f),
            });
            L.Add(new ScatterLayer
            {
                Name = "Wildflowers",
                Prefabs = FindPrefabs(MeadowRoot,
                    "SM_Env_Wildflowers_01", "SM_Env_Wildflowers_02", "SM_Env_Wildflowers_03",
                    "SM_Env_Flowers_Flat_01", "SM_Env_Flowers_Flat_02", "SM_Env_Flowers_Flat_03",
                    "SM_Env_Wildflowers_Patch_01", "SM_Env_Wildflowers_Patch_02", "SM_Env_Wildflowers_Patch_03"),
                MinHeight = 3f, MaxHeight = 26f, MaxSlopeDeg = 25f,
                AllowedBiomes = { Biome.Meadow }, WaterNeed = 0.55f,
                DensityPerSqm = 0.008f, ForestSuppression = 0.75f,
                ScaleRange = new Vector2(0.9f, 1.4f),
            });
            L.Add(new ScatterLayer
            {
                Name = "GroundCover",
                Prefabs = FindPrefabs(MeadowRoot,
                    "SM_Env_Ground_Cover_01", "SM_Env_Ground_Cover_02", "SM_Env_Ground_Cover_03"),
                MinHeight = 2.5f, MaxHeight = 34f, MaxSlopeDeg = 30f,
                AllowedBiomes = { Biome.Meadow, Biome.ForestFloor },
                WaterNeed = 0.55f, DensityPerSqm = 0.025f,
                ScaleRange = new Vector2(0.9f, 1.6f),
            });

            // ---- Scrub / dry side (Arid pack) ----
            L.Add(new ScatterLayer
            {
                Name = "ScrubDeadTrees",
                Prefabs = FindPrefabs(AridRoot, "SM_Env_Tree_Dead_01", "SM_Env_Tree_Dead_02",
                    "SM_Env_Bush_Bramble_01", "SM_Env_Bush_Bramble_02"),
                MinHeight = 3f, MaxHeight = 40f, MaxSlopeDeg = 30f,
                AllowedBiomes = { Biome.Scrub }, WaterNeed = 0.12f,
                DensityPerSqm = 0.003f, CanopyRadius = 2.2f,
                ScaleRange = new Vector2(0.85f, 1.4f),
            });
            L.Add(new ScatterLayer
            {
                Name = "CactiSucculents",
                Prefabs = FindPrefabs(AridRoot, "SM_Env_Cactus_01", "SM_Env_Cactus_02", "SM_Env_Cactus_03",
                    "SM_Env_Succulent_01", "SM_Env_Succulent_02", "SM_Env_Succulent_03",
                    "SM_Env_Succulent_04", "SM_Env_Succulent_05", "SM_Env_Succulent_06"),
                MinHeight = 3f, MaxHeight = 36f, MaxSlopeDeg = 28f,
                AllowedBiomes = { Biome.Scrub }, WaterNeed = 0.06f,
                DensityPerSqm = 0.006f, CanopyRadius = 0.8f,
                ScaleRange = new Vector2(0.8f, 1.5f),
            });

            // ---- Wetland (Swamp pack) ----
            L.Add(new ScatterLayer
            {
                Name = "WetlandReeds",
                Prefabs = FindPrefabs(SwampRoot, "SM_Env_Reeds_01", "SM_Env_Reeds_02",
                    "SM_Env_SwampGrass_Group_01", "SM_Env_SwampGrass_Group_02", "SM_Env_SwampGrass_Group_03",
                    "SM_Env_Grass_Swamp_01", "SM_Env_Grass_Swamp_02"),
                MinHeight = 0.4f, MaxHeight = 10f, MaxSlopeDeg = 22f,
                AllowedBiomes = { Biome.Wetland }, WaterNeed = 0.92f,
                MaxWaterDepth = 0.5f,
                DensityPerSqm = 0.07f,
                ScaleRange = new Vector2(0.9f, 1.5f),
            });
            L.Add(new ScatterLayer
            {
                Name = "WetlandMoss",
                Prefabs = FindPrefabs(SwampRoot, "SM_Env_MossMound_01", "SM_Env_MossMound_02", "SM_Env_MossMound_03"),
                MinHeight = 0.4f, MaxHeight = 9f, MaxSlopeDeg = 20f,
                AllowedBiomes = { Biome.Wetland }, WaterNeed = 0.85f,
                MaxWaterDepth = 0.15f,
                DensityPerSqm = 0.006f, CanopyRadius = 0.9f,
                ScaleRange = new Vector2(0.8f, 1.4f),
            });

            // ---- Alpine (Alpine pack) ----
            L.Add(new ScatterLayer
            {
                Name = "AlpineCover",
                Prefabs = FindPrefabs(AlpineRoot, "SM_Env_GroundCover_01", "SM_Env_GroundCover_02",
                    "SM_Env_GroundCover_03", "SM_Env_Grass_01", "SM_Env_Moss_Lumps_01", "SM_Env_Moss_Lumps_02",
                    "SM_Env_Bush_01_Alt", "SM_Env_Bush_02_Alt", "SM_Env_Bush_Flower_01_Alt"),
                MinHeight = 18f, MaxHeight = 56f, MaxSlopeDeg = 32f,
                AllowedBiomes = { Biome.AlpineMeadow, Biome.Rock },
                DensityPerSqm = 0.05f,
                ScaleRange = new Vector2(0.9f, 1.5f),
            });
            L.Add(new ScatterLayer
            {
                Name = "AlpinePines",
                Prefabs = FindPrefabs(AlpineRoot, "SM_Env_Pine_01", "SM_Env_Pine_02", "SM_Env_Pine_03",
                    "SM_Env_Pine_04", "SM_Env_Pine_05"),
                MinHeight = 52f, MaxHeight = 74f, MaxSlopeDeg = 36f,
                AllowedBiomes = { Biome.Rock, Biome.Snow, Biome.AlpineMeadow },
                DensityPerSqm = 0.012f, CanopyRadius = 2.6f, YStretch = 0.18f,
                ScaleRange = new Vector2(0.9f, 1.5f),
            });
            L.Add(new ScatterLayer
            {
                Name = "BarePines",
                Prefabs = FindPrefabs(AlpineRoot, "SM_Env_Pine_NoLeaves_01", "SM_Env_Pine_Stump_01"),
                MinHeight = 36f, MaxHeight = 54f, MaxSlopeDeg = 34f,
                AllowedBiomes = { Biome.AlpineMeadow, Biome.Rock, Biome.Meadow },
                DensityPerSqm = 0.004f, CanopyRadius = 2f,
                ScaleRange = new Vector2(0.9f, 1.4f),
            });

            // ---- Rocks, biome-correct ----
            L.Add(new ScatterLayer
            {
                Name = "MeadowRocks",
                Prefabs = FindPrefabs(MeadowRoot, "SM_Env_Rock_01", "SM_Env_Rock_02", "SM_Env_Rock_03",
                    "SM_Env_Rock_Small_01", "SM_Env_Rock_Round_01"),
                MinHeight = 3f, MaxHeight = 44f, MaxSlopeDeg = 45f,
                AllowedBiomes = { Biome.Meadow, Biome.ForestFloor, Biome.Wetland },
                DensityPerSqm = 0.0012f, CanopyRadius = 1.1f, MaxTiltDeg = 7f,
                ScaleRange = new Vector2(0.7f, 1.6f),
                IsVegetation = false,
            });
            L.Add(new ScatterLayer
            {
                Name = "DryRocks",
                Prefabs = FindPrefabs(AridRoot, "SM_Env_Rock_01", "SM_Env_Rock_02", "SM_Env_Rock_03",
                    "SM_Env_Rock_04", "SM_Env_Rock_05", "SM_Env_Rock_06", "SM_Env_Rock_07",
                    "SM_Env_Rock_08", "SM_Env_Rock_09", "SM_Env_Rock_10", "SM_Env_Rock_11"),
                MinHeight = 3f, MaxHeight = 60f, MaxSlopeDeg = 50f,
                AllowedBiomes = { Biome.Scrub, Biome.Rock, Biome.AlpineMeadow },
                PreferRocky = true, DensityPerSqm = 0.005f, CanopyRadius = 1.3f, MaxTiltDeg = 7f,
                ScaleRange = new Vector2(0.7f, 1.8f),
                IsVegetation = false,
            });
            L.Add(new ScatterLayer
            {
                Name = "CliffRocks",
                Prefabs = FindPrefabs(MeadowRoot, "SM_Env_Rock_Pile_01", "SM_Env_Rock_Pile_02", "SM_Env_Rock_Cliff_01"),
                MinHeight = 20f, MaxHeight = 48f, MaxSlopeDeg = 60f,
                PreferRocky = true, DensityPerSqm = 0.0012f, CanopyRadius = 2.4f,
                ScaleRange = new Vector2(0.9f, 1.8f),
                IsVegetation = false,
            });

            L.Add(new ScatterLayer
            {
                Name = "DryGrass",
                Prefabs = FindPrefabs(AridRoot, "SM_Env_GroundCover_01", "SM_Env_GroundCover_02",
                    "SM_Env_GroundCover_03", "SM_Env_Ground_Mound_Small_01"),
                MinHeight = 3f, MaxHeight = 42f, MaxSlopeDeg = 32f,
                AllowedBiomes = { Biome.Scrub }, WaterNeed = 0.15f,
                DensityPerSqm = 0.035f,
                ScaleRange = new Vector2(0.9f, 1.6f),
            });
            L.Add(new ScatterLayer
            {
                Name = "CragsMeadow",
                Prefabs = FindPrefabs(MeadowRoot, "SM_Env_Ground_Cliff_Large_01", "SM_Env_Ground_Cliff_Large_02",
                    "SM_Env_Rock_Cliff_02", "SM_Env_Rock_Cliff_03"),
                MinHeight = 4f, MaxHeight = 58f, MaxSlopeDeg = 90f,
                AllowedBiomes = { Biome.Meadow, Biome.ForestFloor, Biome.AlpineMeadow },
                PreferRocky = true, DensityPerSqm = 0.0012f,
                CanopyRadius = 6f, YOffset = -0.35f, MaxTiltDeg = 4f,
                ScaleRange = new Vector2(1.3f, 2.6f),
                IsVegetation = false,
            });
            L.Add(new ScatterLayer
            {
                Name = "CragsDry",
                Prefabs = FindPrefabs(AridRoot, "SM_Env_Rock_12", "SM_Env_Rock_13", "SM_Env_Rock_14",
                    "SM_Env_Rock_Arch_01"),
                MinHeight = 4f, MaxHeight = 62f, MaxSlopeDeg = 90f,
                AllowedBiomes = { Biome.Scrub, Biome.Rock },
                PreferRocky = true, DensityPerSqm = 0.0009f,
                CanopyRadius = 7f, YOffset = -0.35f, MaxTiltDeg = 4f,
                ScaleRange = new Vector2(1.4f, 2.8f),
                IsVegetation = false,
            });

            L.Add(new ScatterLayer
            {
                Name = "ForestFerns",
                Prefabs = FindPrefabs(TropicalRoot, "SM_Env_Fern_01", "SM_Env_Fern_02",
                    "SM_Env_Fern_03", "SM_Env_Fern_04"),
                MinHeight = 3f, MaxHeight = 34f, MaxSlopeDeg = 30f,
                AllowedBiomes = { Biome.ForestFloor }, WaterNeed = 0.6f,
                DensityPerSqm = 0.03f, MaxTiltDeg = 3f,
                ScaleRange = new Vector2(0.8f, 1.5f),
            });
            L.Add(new ScatterLayer
            {
                Name = "WetlandTrees",
                Prefabs = FindPrefabs(SwampRoot, "SM_Env_Tree_Mangrove_01", "SM_Env_Tree_Mangrove_02",
                    "SM_Env_Tree_Mangrove_03", "SM_Env_Swamp_Tree_Root_Preset_01", "SM_Env_Swamp_Tree_Root_Preset_02"),
                MinHeight = 0.5f, MaxHeight = 10f, MaxSlopeDeg = 22f,
                AllowedBiomes = { Biome.Wetland }, WaterNeed = 0.9f,
                MaxWaterDepth = 0.35f,
                DensityPerSqm = 0.006f, CanopyRadius = 3f, YStretch = 0.16f,
                ScaleRange = new Vector2(0.9f, 1.4f),
            });
            L.Add(new ScatterLayer
            {
                Name = "SwampLogs",
                Prefabs = FindPrefabs(SwampRoot, "SM_Env_Swamp_Log_01", "SM_Env_Swamp_Log_02"),
                MinHeight = 0.5f, MaxHeight = 20f, MaxSlopeDeg = 25f,
                AllowedBiomes = { Biome.Wetland, Biome.ForestFloor },
                DensityPerSqm = 0.0012f, CanopyRadius = 1.5f, MaxTiltDeg = 6f,
                ScaleRange = new Vector2(0.9f, 1.4f),
            });
            L.Add(new ScatterLayer
            {
                Name = "CoastCliffWall",
                Prefabs = FindPrefabs(MeadowRoot, "SM_Env_Ground_Cliff_Large_01", "SM_Env_Ground_Cliff_Large_02"),
                CoastalWall = true, YOffset = -0.15f,
                ScaleRange = new Vector2(0.9f, 1.2f),
                IsVegetation = false,
            });

            foreach (var wall in BuildCliffWallLayers())
                L.Add(wall);

            AssetDatabase.CreateAsset(cfg, ScatterConfigPath);
            AssetDatabase.SaveAssets();
            return cfg;
        }

        [MenuItem("AniML/World/Export Biome Map")]
        public static void ExportBiomeMap()
        {
            var world = Object.FindFirstObjectByType<ProceduralWorld>();
            if (world == null || world.Settings == null)
            {
                Debug.LogError("[AniML] No configured ProceduralWorld in the open scene.");
                return;
            }
            world.EnsureSamplers();

            const int res = 512;
            float size = world.Settings.WorldSize;
            var tex = new Texture2D(res, res, TextureFormat.RGB24, false);
            for (int py = 0; py < res; py++)
            {
                for (int px = 0; px < res; px++)
                {
                    float x = (px / (float)(res - 1) - 0.5f) * size;
                    float z = (py / (float)(res - 1) - 0.5f) * size;
                    tex.SetPixel(px, py, BiomeDebugColor(world.SampleBiome(x, z)));
                }
            }
            tex.Apply();

            string dir = System.Environment.GetEnvironmentVariable("ANIML_SHOT_DIR");
            if (string.IsNullOrEmpty(dir)) dir = Path.Combine(Path.GetTempPath(), "animl_world_shots");
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, "biome_map.png");
            File.WriteAllBytes(path, tex.EncodeToPNG());
            Object.DestroyImmediate(tex);
            Debug.Log($"[AniML] Biome map written to {path}");
        }

        private static Color BiomeDebugColor(Biome b) => b switch
        {
            Biome.Ocean => new Color(0.15f, 0.3f, 0.55f),
            Biome.Beach => new Color(0.92f, 0.85f, 0.6f),
            Biome.Wetland => new Color(0.1f, 0.55f, 0.5f),
            Biome.Meadow => new Color(0.5f, 0.75f, 0.35f),
            Biome.Scrub => new Color(0.8f, 0.65f, 0.3f),
            Biome.ForestFloor => new Color(0.12f, 0.45f, 0.15f),
            Biome.AlpineMeadow => new Color(0.55f, 0.7f, 0.55f),
            Biome.Rock => new Color(0.5f, 0.48f, 0.46f),
            Biome.Snow => Color.white,
            Biome.RiverChannel => new Color(0.3f, 0.6f, 0.9f),
            _ => Color.magenta,
        };

        private const string MeadowRoot = "Assets/Synty/PolygonNatureBiomes/PNB_Meadow_Forest";
        private const string AridRoot = "Assets/Synty/PolygonNatureBiomes/PNB_Arid_Desert";
        private const string SwampRoot = "Assets/Synty/PolygonNatureBiomes/PNB_Swamp_Marshland";
        private const string TropicalRoot = "Assets/Synty/PolygonNatureBiomes/PNB_Tropical_Jungle";
        private const string AlpineRoot = "Assets/Synty/PolygonNatureBiomes/PNB_Alpine_Mountain";

        /// <summary>Resolve prefabs by exact name within ONE pack — the same
        /// names exist in several packs with very different materials.</summary>
        private static System.Collections.Generic.List<GameObject> FindPrefabs(string searchRoot, params string[] names)
        {
            var list = new System.Collections.Generic.List<GameObject>();
            foreach (var name in names)
            {
                var guids = AssetDatabase.FindAssets($"{name} t:Prefab", new[] { searchRoot });
                GameObject found = null;
                foreach (var guid in guids)
                {
                    var path = AssetDatabase.GUIDToAssetPath(guid);
                    if (Path.GetFileNameWithoutExtension(path) == name)
                    {
                        found = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                        break;
                    }
                }
                if (found != null) list.Add(found);
                else Debug.LogWarning($"[AniML] Scatter prefab not found in {searchRoot}: {name}");
            }
            return list;
        }

        private static WorldSettings LoadOrCreateSettings()
        {
            var s = AssetDatabase.LoadAssetAtPath<WorldSettings>(SettingsPath);
            if (s == null)
            {
                s = ScriptableObject.CreateInstance<WorldSettings>();
                AssetDatabase.CreateAsset(s, SettingsPath);
                AssetDatabase.SaveAssets();
            }
            return s;
        }

        private static Material LoadOrCreateTerrainMaterial()
        {
            var m = AssetDatabase.LoadAssetAtPath<Material>(MaterialPath);
            if (m == null)
            {
                var shader = Shader.Find("AniML/TerrainVertexColor");
                if (shader == null)
                {
                    Debug.LogError("[AniML] TerrainVertexColor shader not found — falling back to URP Lit (no vertex color).");
                    shader = Shader.Find("Universal Render Pipeline/Lit");
                }
                m = new Material(shader) { name = "M_Terrain" };
                AssetDatabase.CreateAsset(m, MaterialPath);
                AssetDatabase.SaveAssets();
            }
            // Art direction: stylized, not cartoonish — smooth lighting to
            // match the Synty packs' own look (banding available via slider).
            if (m.HasProperty("_ToonSteps"))
                m.SetFloat("_ToonSteps", 0f);

            // Ground detail: Synty's own tiling terrain textures, blended by
            // the mesh splat weights.
            AssignTexture(m, "_GrassTex", MeadowRoot + "/Terrain", "Grass_Texture_01");
            AssignTexture(m, "_DirtTex", MeadowRoot + "/Terrain", "Dirt_Texture_01");
            AssignTexture(m, "_RockTex", MeadowRoot + "/Terrain", "Gravel_Texture_01");
            AssignTexture(m, "_SnowTex", AlpineRoot, "Snow_01");
            EditorUtility.SetDirty(m);
            return m;
        }

        /// <summary>
        /// Synty's foliage shader ships per-instance colour-variation dials
        /// (used in their Alpine materials, zeroed in the others). Turn them
        /// on everywhere so no two trees render the same hue.
        /// </summary>
        private static void EnableFoliageColourVariation()
        {
            var guids = AssetDatabase.FindAssets("t:Material", new[] { "Assets/Synty" });
            int touched = 0;
            foreach (var guid in guids)
            {
                var mat = AssetDatabase.LoadAssetAtPath<Material>(AssetDatabase.GUIDToAssetPath(guid));
                if (mat == null || mat.shader == null || !mat.shader.name.Contains("Foliage"))
                    continue;
                if (mat.HasProperty("_ColourNoiseLargeScale") && mat.GetFloat("_ColourNoiseLargeScale") < 0.01f)
                {
                    mat.SetFloat("_ColourNoiseLargeScale", 0.37f);   // Synty's own Alpine values
                    mat.SetFloat("_ColourNoiseSmallScale", 0.55f);
                    if (mat.HasProperty("_ColourNoiseScale") && mat.GetFloat("_ColourNoiseScale") < 0.01f)
                        mat.SetFloat("_ColourNoiseScale", 0.11f);
                    EditorUtility.SetDirty(mat);
                    touched++;
                }
            }
            if (touched > 0)
            {
                AssetDatabase.SaveAssets();
                Debug.Log($"[AniML] Enabled colour variation on {touched} foliage materials.");
            }
        }

        /// <summary>
        /// Wire the wonder prefab slots from the packs. AssetDatabase search
        /// is editor-only; the placer itself is a runtime component, so the
        /// wonders get built in Play mode too, not just in editor captures.
        /// </summary>
        private static void WireWonderPrefabs(WonderPlacer placer)
        {
            void Wire(string key, string packRoot, string prefabName)
            {
                var list = FindPrefabs(packRoot, prefabName);
                if (list.Count > 0)
                    placer.SetSlot(key, list[0]);
                else
                    Debug.LogWarning($"[AniML] Wonder prefab not found: {prefabName} under {packRoot}");
            }

            Wire("Waterfall", TropicalRoot, "SM_Waterfall_Mesh_01");
            Wire("WaterfallFX", TropicalRoot, "FX_Waterfall_Impact_01");
            Wire("GiantTree", MeadowRoot, "SM_Env_Tree_Meadow_01");
            Wire("GiantTreeRockA", MeadowRoot, "SM_Env_Rock_Small_01");
            Wire("GiantTreeRockB", MeadowRoot, "SM_Env_Rock_Round_01");
            Wire("SulphurGround", AridRoot, "SM_Env_Sulphur_Ground_01");
            Wire("SulphurPoolA", AridRoot, "SM_Env_Sulphur_Pool_01");
            Wire("SulphurPoolB", AridRoot, "SM_Env_Sulphur_Pool_03");
            Wire("SulphurPoolC", AridRoot, "SM_Env_Sulphur_Pool_04");
            Wire("SulphurSmoke", AridRoot, "FX_Sulphur_Smoke");
            Wire("WhaleBones", TropicalRoot, "SM_Prop_Whale_Skeleton_01");
            Wire("SummitRangeA", AlpineRoot, "SM_Env_MountainRange_01");
            Wire("SummitRangeB", AlpineRoot, "SM_Env_MountainRange_02");
            Wire("SummitRangeC", AlpineRoot, "SM_Env_MountainRange_03");
            EditorUtility.SetDirty(placer);
        }

        private static void AssignTexture(Material m, string property, string searchRoot, string name)
        {
            var guids = AssetDatabase.FindAssets($"{name} t:Texture2D", new[] { searchRoot });
            foreach (var guid in guids)
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                if (Path.GetFileNameWithoutExtension(path) == name)
                {
                    m.SetTexture(property, AssetDatabase.LoadAssetAtPath<Texture2D>(path));
                    return;
                }
            }
            Debug.LogWarning($"[AniML] Terrain texture not found: {name} under {searchRoot} (using white).");
        }

        private static void EnsureWaterPlane(WorldSettings settings)
        {
            var water = GameObject.Find("Water");
            if (water == null)
            {
                water = GameObject.CreatePrimitive(PrimitiveType.Plane);
                water.name = "Water";
                Object.DestroyImmediate(water.GetComponent<Collider>());
            }

            water.transform.position = new Vector3(0f, settings.SeaLevel, 0f);
            // Unity plane primitive is 10x10 m at scale 1. Extend far past the
            // island so the horizon is ocean, not skybox floor (until the real
            // sky system exists).
            float scale = settings.WorldSize / 10f * 8f;
            water.transform.localScale = new Vector3(scale, 1f, scale);

            water.GetComponent<MeshRenderer>().sharedMaterial = EnsureWaterMaterial();
        }

        private static Material EnsureWaterMaterial()
        {
            var mat = AssetDatabase.LoadAssetAtPath<Material>(WaterMaterialPath);
            if (mat == null)
            {
                mat = new Material(Shader.Find("Universal Render Pipeline/Lit")) { name = "M_Water_Placeholder" };
                mat.SetFloat("_Surface", 1f); // transparent
                mat.SetFloat("_Blend", 0f);
                mat.SetOverrideTag("RenderType", "Transparent");
                mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                mat.SetInt("_ZWrite", 0);
                mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
                mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                mat.SetColor("_BaseColor", new Color(0.25f, 0.5f, 0.62f, 0.72f));
                mat.SetFloat("_Smoothness", 0.85f);
                AssetDatabase.CreateAsset(mat, WaterMaterialPath);
                AssetDatabase.SaveAssets();
            }
            return mat;
        }
    }
}
