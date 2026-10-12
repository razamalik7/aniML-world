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

            // Batch runs can pick the landform profile and seed via env vars.
            var profEnv = System.Environment.GetEnvironmentVariable("ANIML_PROFILE");
            if (!string.IsNullOrEmpty(profEnv)
                && System.Enum.TryParse(profEnv, true, out LandformProfile prof)
                && settings.Profile != prof)
            {
                settings.Profile = prof;
                EditorUtility.SetDirty(settings);
                AssetDatabase.SaveAssets();
            }
            var seedEnv = System.Environment.GetEnvironmentVariable("ANIML_SEED");
            if (!string.IsNullOrEmpty(seedEnv)
                && int.TryParse(seedEnv, out int seed)
                && settings.Seed != seed)
            {
                settings.Seed = seed;
                EditorUtility.SetDirty(settings);
                AssetDatabase.SaveAssets();
            }

            var material = LoadOrCreateTerrainMaterial();
            if (material.HasProperty("_SeaLevel"))
            {
                material.SetFloat("_SeaLevel", settings.SeaLevel);
                EditorUtility.SetDirty(material);
            }

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

            world.Configure(settings, material, EnsureWaterMaterial(),
                EnsureRiverWaterMaterial(), EnsureLakeWaterMaterial());
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

            var ambient = root.GetComponent<AmbientFX>();
            if (ambient == null)
                ambient = root.AddComponent<AmbientFX>();
            WireAmbientFX(ambient);
            ambient.Place(world);

            // Day/night + weather. The sun is the scene's directional light.
            var sunGo = GameObject.Find("Directional Light");
            var sky = root.GetComponent<SkySystem>();
            if (sky == null)
                sky = root.AddComponent<SkySystem>();
            if (sunGo != null)
            {
                var sunLight = sunGo.GetComponent<Light>();
                sky.Configure(sunLight);
                // Cloud shadows: a soft dapple cookie that drifts in Play.
                sunLight.cookie = EnsureCloudCookie();
                var lightData = sunGo.GetComponent<UnityEngine.Rendering.Universal.UniversalAdditionalLightData>();
                if (lightData == null)
                    lightData = sunGo.AddComponent<UnityEngine.Rendering.Universal.UniversalAdditionalLightData>();
                lightData.lightCookieSize = new Vector2(430f, 430f);
            }
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.Exponential;
            RenderSettings.fogDensity = 0.0005f;   // subtle depth cue only
            sky.Apply();

            if (root.GetComponent<UnderwaterTint>() == null)
                root.AddComponent<UnderwaterTint>();
            if (root.GetComponent<VolcanoPulse>() == null)
                root.AddComponent<VolcanoPulse>();

            var wildfire = root.GetComponent<WildfireSystem>();
            if (wildfire == null)
                wildfire = root.AddComponent<WildfireSystem>();
            void WireFire(string key, string packRoot, string prefabName)
            {
                var list = FindPrefabs(packRoot, prefabName);
                if (list.Count > 0)
                    wildfire.SetSlot(key, list[0]);
            }
            WireFire("Fire", GenericRoot, "FX_Fire_01");
            WireFire("Smoke", GenericRoot, "FX_Smoke_01");
            WireFire("Lightning", AridRoot, "FX_Electricity_01");
            EditorUtility.SetDirty(wildfire);

            var weather = root.GetComponent<WeatherSystem>();
            if (weather == null)
                weather = root.AddComponent<WeatherSystem>();
            weather.SetSlot("Rain", LoadCorePrefab("FX_Rain_01"));
            weather.SetSlot("Snow", LoadCorePrefab("FX_Snow_01"));
            EditorUtility.SetDirty(weather);

            EnableFoliageColourVariation();
            EnableFoliageWind();
            EnsurePostProcessing();
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
                CaptureFeatureShots(worldComp, outDir, "");

            // Time-of-day variants: golden hour and night under the moon.
            var sky = Object.FindFirstObjectByType<SkySystem>();
            if (sky != null)
            {
                float orig = sky.TimeOfDay;
                sky.TimeOfDay = 17.4f;   // golden hour, sun still up
                sky.Apply();
                Capture(new Vector3(-380f, 95f, -420f), Quaternion.Euler(14f, 38f, 0f), 55f,
                    Path.Combine(outDir, "world_sunset.png"));
                sky.TimeOfDay = 23.6f;
                sky.Apply();
                Capture(new Vector3(-380f, 95f, -420f), Quaternion.Euler(14f, 38f, 0f), 55f,
                    Path.Combine(outDir, "world_night.png"));
                sky.TimeOfDay = orig;
                sky.Apply();
            }

            ExportBiomeMap();
            DumpLandformMeshBounds();
            Debug.Log($"[AniML] Wrote verification renders to {outDir}");
        }

        /// <summary>
        /// The landmark close-up reel: every wonder with its bespoke
        /// framing, the primary summit, the canyon crossing, and a ford.
        /// Shared by verification bakes (no prefix) and art-direction
        /// look comparisons (prefixed per look).
        /// </summary>
        private static void CaptureFeatureShots(ProceduralWorld worldComp, string outDir, string prefix)
        {
            // No camera ever sits under the terrain: rerouted rivers and
            // fresh crags keep swallowing fixed offsets, and a buried eye
            // renders see-through backface garbage.
            Vector3 AboveGround(Vector3 eye)
            {
                eye.y = Mathf.Max(eye.y, worldComp.SampleHeight(eye.x, eye.z) + 6f);
                return eye;
            }

            // Terrain must not rise across the sight line (crags can still
            // photobomb, but ridge-blocked shots are the common failure).
            bool Clear(Vector3 eye, Vector3 target)
            {
                for (int s = 2; s <= 11; s++)
                {
                    float t = s / 12f;
                    var p = Vector3.Lerp(eye, target, t);
                    if (worldComp.SampleHeight(p.x, p.z) > p.y - 0.5f)
                        return false;
                }
                return true;
            }

            // Keep the intended framing when it works; orbit the target for
            // an unblocked azimuth when it doesn't.
            Vector3 FindClearEye(Vector3 preferredEye, Vector3 target)
            {
                var flat = new Vector2(preferredEye.x - target.x, preferredEye.z - target.z);
                float dist = Mathf.Max(flat.magnitude, 12f);
                float lift = preferredEye.y - target.y;
                float baseAz = Mathf.Atan2(flat.x, flat.y);
                for (int k = 0; k < 8; k++)
                {
                    float az = baseAz + k * Mathf.PI / 4f;
                    var eye = AboveGround(target + new Vector3(
                        Mathf.Sin(az) * dist, lift, Mathf.Cos(az) * dist));
                    if (Clear(eye, target))
                        return eye;
                }
                // Every azimuth blocked — climb until the sight line opens.
                var high = AboveGround(preferredEye);
                for (int k = 0; k < 5 && !Clear(high, target); k++)
                    high.y += 18f;
                return high;
            }

            foreach (var site in worldComp.Wonders)
            {
                float rad = site.YawDeg * Mathf.Deg2Rad;
                var facing = new Vector3(Mathf.Sin(rad), 0f, Mathf.Cos(rad));
                Vector3 eye;
                Vector3 target = site.Pos + Vector3.up * 5f;
                switch (site.Kind)
                {
                    case "Waterfall":
                        // Stand downstream, look back up at the falls.
                        eye = site.Pos + facing * 42f + Vector3.up * 12f;
                        break;
                    case "Cave":
                        // Stand on the approach, look into the mouth.
                        eye = site.Pos + facing * 40f + Vector3.up * 12f;
                        target = site.Pos + Vector3.up * 3f;
                        break;
                    case "RedArch":
                        // Hover beside the rim, arch in frame, slot behind.
                        eye = site.Pos - facing * 26f + Vector3.up * 14f;
                        target = site.Pos + facing * 20f + Vector3.down * 6f;
                        break;
                    case "LavaCrater":
                        // From above the rim, looking down into the lake.
                        eye = site.Pos + new Vector3(-40f, 42f, -40f);
                        target = site.Pos;
                        break;
                    default:
                        eye = site.Pos + new Vector3(34f, 26f, -34f);
                        break;
                }
                eye = FindClearEye(eye, target);
                Capture(eye, Quaternion.LookRotation(target - eye), 50f,
                    Path.Combine(outDir, $"{prefix}wonder_{site.Kind}.png"));
            }
            if (worldComp.Summits.Count > 0)
            {
                var s = worldComp.Summits[0];
                var eye = AboveGround(s.Pos + new Vector3(-160f, 40f, -160f));
                Capture(eye, Quaternion.LookRotation(s.Pos + Vector3.up * 20f - eye), 55f,
                    Path.Combine(outDir, $"{prefix}summit.png"));
            }
            if (worldComp.CanyonCrossing is Vector2 cc)
            {
                // Shoot along the corridor axis from above one rim, so the
                // ramp notch, the channel, and the far rim stack in frame.
                float cy = worldComp.SampleHeight(cc.x, cc.y);
                var target = new Vector3(cc.x, cy + 4f, cc.y);
                var d2 = worldComp.CanyonCrossingDir ?? Vector2.right;
                var axis = new Vector3(d2.x, 0f, d2.y);
                var eye = AboveGround(target + axis * 75f + Vector3.up * 48f);
                Capture(eye, Quaternion.LookRotation(target - eye), 50f,
                    Path.Combine(outDir, $"{prefix}canyon_crossing.png"));
            }
            if (worldComp.Fords.Count > 0)
            {
                // Widest ford = the lowland trunk-river crossing, the one
                // that actually photographs as a gravel bar.
                var f = worldComp.Fords[0];
                foreach (var candidate in worldComp.Fords)
                    if (candidate.Radius > f.Radius)
                        f = candidate;
                float fy = worldComp.SampleHeight(f.Pos.x, f.Pos.y);
                var target = new Vector3(f.Pos.x, fy + 1.5f, f.Pos.y);
                var eye = AboveGround(target + new Vector3(34f, 26f, -34f));
                Capture(eye, Quaternion.LookRotation(target - eye), 50f,
                    Path.Combine(outDir, $"{prefix}ford.png"));
            }
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

        /// <summary>
        /// Batchmode entry: render the SAME three scenes through each
        /// candidate art-direction grade so the look can be chosen from
        /// photos, not adjectives. Nothing is saved — the temporary volume
        /// profiles live in memory and the scene is left untouched on disk.
        /// </summary>
        public static void CaptureLooks()
        {
            GenerateInScene();

            string outDir = System.Environment.GetEnvironmentVariable("ANIML_SHOT_DIR");
            if (string.IsNullOrEmpty(outDir))
                outDir = Path.Combine(Path.GetTempPath(), "animl_look_shots");
            Directory.CreateDirectory(outDir);

            // Consistent light: mid-afternoon for every look, so the grades
            // (not the sun position) are what differ between photos.
            var sky = Object.FindFirstObjectByType<SkySystem>();
            if (sky != null) { sky.TimeOfDay = 15.0f; sky.Apply(); }

            var volGo = GameObject.Find("PostVolume");
            var vol = volGo != null ? volGo.GetComponent<UnityEngine.Rendering.Volume>() : null;
            var worldComp = Object.FindFirstObjectByType<ProceduralWorld>();

            void Shoot(string look)
            {
                Capture(new Vector3(-380f, 95f, -420f), Quaternion.Euler(14f, 38f, 0f), 55f,
                    Path.Combine(outDir, $"{look}_vista.png"));
                var eye = new Vector3(280f, 175f, -180f);
                var target = new Vector3(150f, 10f, -50f);
                Capture(eye, Quaternion.LookRotation(target - eye), 48f,
                    Path.Combine(outDir, $"{look}_godview.png"));
                // Full landmark reel per look — grades are judged on real
                // subjects (water, red rock, forest, snow), not one vista.
                if (worldComp != null)
                    CaptureFeatureShots(worldComp, outDir, $"{look}_");
            }

            UnityEngine.Rendering.VolumeProfile Fresh()
            {
                var p = ScriptableObject.CreateInstance<UnityEngine.Rendering.VolumeProfile>();
                var bloom = p.Add<UnityEngine.Rendering.Universal.Bloom>(false);
                bloom.intensity.Override(0.55f);
                bloom.threshold.Override(1.0f);
                bloom.scatter.Override(0.6f);
                var tone = p.Add<UnityEngine.Rendering.Universal.Tonemapping>(false);
                tone.mode.Override(UnityEngine.Rendering.Universal.TonemappingMode.Neutral);
                return p;
            }

            // ---- Look 1: warm painterly (golden key, cool shadows, soft haze)
            {
                var p = Fresh();
                var grade = p.Add<UnityEngine.Rendering.Universal.ColorAdjustments>(false);
                grade.saturation.Override(24f);
                grade.contrast.Override(12f);
                // Warmth comes from the color filter alone — WhiteBalance's
                // LMS math NaNs on extreme HDR snow in batch captures. Bold
                // on purpose: a grade you can't tell apart in a side-by-side
                // is not a grade worth choosing between.
                grade.colorFilter.Override(new Color(1.0f, 0.90f, 0.76f));
                var split = p.Add<UnityEngine.Rendering.Universal.SplitToning>(false);
                split.shadows.Override(new Color(0.38f, 0.48f, 0.66f));
                split.highlights.Override(new Color(1.0f, 0.82f, 0.58f));
                split.balance.Override(-22f);
                if (vol != null) vol.profile = p;
                RenderSettings.fogColor = new Color(0.78f, 0.72f, 0.62f);
                RenderSettings.fogDensity = 0.0007f;
                Shoot("look_painterly");
            }

            // ---- Look 2: storybook diorama (tilt-shift miniature, clear air)
            {
                var p = Fresh();
                var grade = p.Add<UnityEngine.Rendering.Universal.ColorAdjustments>(false);
                grade.saturation.Override(26f);
                grade.contrast.Override(13f);
                // NOTE: the tilt-shift blur itself is a runtime DoF (URP's
                // DepthOfField does not evaluate in manual batch captures —
                // Gaussian and Bokeh both render sharp). Photo previews get
                // a composited graduated blur instead; the in-game look
                // applies DoF live where it works normally.
                var vig = p.Add<UnityEngine.Rendering.Universal.Vignette>(false);
                vig.intensity.Override(0.22f);
                vig.smoothness.Override(0.5f);
                if (vol != null) vol.profile = p;
                RenderSettings.fogColor = new Color(0.72f, 0.76f, 0.80f);
                RenderSettings.fogDensity = 0.0003f;   // miniatures have clear air
                // Overlay decision, not a base grade — two shots suffice
                // (the miniature blur is composited onto these in post).
                Capture(new Vector3(-380f, 95f, -420f), Quaternion.Euler(14f, 38f, 0f), 55f,
                    Path.Combine(outDir, "look_diorama_vista.png"));
                var dEye = new Vector3(280f, 175f, -180f);
                var dTarget = new Vector3(150f, 10f, -50f);
                Capture(dEye, Quaternion.LookRotation(dTarget - dEye), 48f,
                    Path.Combine(outDir, "look_diorama_godview.png"));
            }

            // ---- Look 3: crisp naturalism (honest hues, cool blue distance)
            {
                var p = Fresh();
                var grade = p.Add<UnityEngine.Rendering.Universal.ColorAdjustments>(false);
                grade.saturation.Override(-2f);
                grade.contrast.Override(6f);
                var wb = p.Add<UnityEngine.Rendering.Universal.WhiteBalance>(false);
                wb.temperature.Override(-12f);
                if (vol != null) vol.profile = p;
                RenderSettings.fogColor = new Color(0.62f, 0.70f, 0.80f);
                RenderSettings.fogDensity = 0.0006f;
                Shoot("look_naturalism");
            }

            Debug.Log($"[AniML] Look comparison renders written to {outDir}");
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
            cam.allowHDR = true;
            UnityEngine.Rendering.Universal.CameraExtensions
                .GetUniversalAdditionalCameraData(cam).renderPostProcessing = true;

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

        private const string RedRockMatPath =
            "Assets/Synty/PolygonNatureBiomes/PNB_Arid_Desert/Materials/Rock_Triplanar_Red_01.mat";

        private static ScatterLayer[] BuildCliffWallLayers()
        {
            // NOTE: Alpine's "Rock_Cliff" set is GLACIER ICE (refractive
            // shader) — real granite is the plain Rock / Rock_Rough meshes,
            // scaled up to wall size.
            var granite = CliffWallLayer("CliffWallGranite", CliffFamily.Granite, AlpineRoot,
                new Vector2(1.4f, 2.0f),
                "SM_Env_Rock_Rough_01", "SM_Env_Rock_Rough_02",
                "SM_Env_Rock_01", "SM_Env_Rock_010");
            var red = CliffWallLayer("CliffWallRedRock", CliffFamily.RedRock, AridRoot,
                new Vector2(0.85f, 1.2f),
                "SM_Env_Rock_Cliff_01", "SM_Env_Rock_Cliff_02", "SM_Env_Rock_Cliff_03",
                "SM_Env_Rock_Cliff_04", "SM_Env_Rock_Cliff_05", "SM_Env_Rock_Cliff_06",
                "SM_Env_Rock_Cliff_07", "SM_Env_Rock_Cliff_08", "SM_Env_Rock_Cliff_09",
                "SM_Env_Rock_Cliff_10", "SM_Env_Rock_Cliff_11", "SM_Env_Rock_Cliff_12",
                "SM_Env_Rock_Cliff_13", "SM_Env_Rock_Cliff_14", "SM_Env_Rock_Cliff_15",
                "SM_Env_Rock_Cliff_16");
            // The default arid rock material is sandstone-tan; Synty ships a
            // proper red variant — use it so red rock is actually RED.
            red.OverrideMaterial = AssetDatabase.LoadAssetAtPath<Material>(RedRockMatPath);
            var mossy = CliffWallLayer("CliffWallMossy", CliffFamily.Mossy, MeadowRoot,
                new Vector2(0.85f, 1.2f),
                "SM_Env_Rock_Cliff_01", "SM_Env_Rock_Cliff_02", "SM_Env_Rock_Cliff_03");
            return new[] { granite, red, mossy };
        }

        /// <summary>Aquatic flora: floating pads on lakes, seaweed and coral
        /// in the sunlit sea shallows.</summary>
        private static ScatterLayer[] BuildAquaticLayers() => new[]
        {
            new ScatterLayer
            {
                Name = "LilyPads",
                Prefabs = FindPrefabs(SwampRoot, "SM_Env_LillyPads_01", "SM_Env_LillyPads_02",
                    "SM_Env_LillyPads_03", "SM_Env_LillyPads_04"),
                MinHeight = 0.5f, MaxHeight = 30f, MaxSlopeDeg = 25f,
                OnWaterSurface = true, MinWaterDepth = 0.25f,
                MinTemp01 = 0.35f,
                DensityPerSqm = 0.02f,
                ScaleRange = new Vector2(0.9f, 1.5f),
            },
            new ScatterLayer
            {
                Name = "Seaweed",
                Prefabs = FindPrefabs(TropicalRoot, "SM_Env_Seaweed_01", "SM_Env_Seaweed_02",
                    "SM_Env_Seaweed_03"),
                MinHeight = -3.6f, MaxHeight = -0.6f, MaxSlopeDeg = 30f,
                AllowedBiomes = { Biome.Ocean },
                MinTemp01 = 0.4f,
                DensityPerSqm = 0.012f,
                ScaleRange = new Vector2(0.9f, 1.6f),
            },
            new ScatterLayer
            {
                Name = "TideRocks",
                Prefabs = FindPrefabs(MeadowRoot, "SM_Env_Rock_Small_01", "SM_Env_Rock_Small_Pile_01",
                    "SM_Env_Rock_Small_Pile_02", "SM_Env_Rock_Round_01"),
                MinHeight = -0.9f, MaxHeight = 0.4f, MaxSlopeDeg = 20f,
                AllowedBiomes = { Biome.Ocean, Biome.Beach },
                DensityPerSqm = 0.004f, MaxTiltDeg = 8f, YOffset = -0.1f,
                ScaleRange = new Vector2(0.7f, 1.4f),
                IsVegetation = false,
            },
            new ScatterLayer
            {
                Name = "Coral",
                Prefabs = FindPrefabs(TropicalRoot, "SM_Prop_Coral_01", "SM_Prop_Coral_03",
                    "SM_Prop_Coral_05", "SM_Prop_Coral_07", "SM_Prop_Coral_09", "SM_Prop_Coral_11"),
                MinHeight = -3.2f, MaxHeight = -0.9f, MaxSlopeDeg = 26f,
                AllowedBiomes = { Biome.Ocean },
                MinTemp01 = 0.55f, YOffset = -0.05f,
                DensityPerSqm = 0.008f, CanopyRadius = 0.8f,
                ScaleRange = new Vector2(0.8f, 1.5f),
                IsVegetation = false,
            },
        };

        /// <summary>
        /// Idempotent migration for pre-existing config assets: wetland
        /// waders stand in the shallows (serialized MaxWaterDepth defaults
        /// to 0 = dry ground only), and layer sets added after the asset was
        /// created (cliff walls, aquatic flora) are appended or refreshed.
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
            // The builders are the source of truth for these layer sets:
            // replace in place so corrections propagate to existing assets.
            foreach (var layer in BuildCliffWallLayers())
            {
                int idx = cfg.Layers.FindIndex(l => l.Name == layer.Name);
                if (idx >= 0)
                    cfg.Layers[idx] = layer;
                else
                    cfg.Layers.Add(layer);
                dirty = true;
            }
            foreach (var layer in BuildAquaticLayers())
            {
                int idx = cfg.Layers.FindIndex(l => l.Name == layer.Name);
                if (idx >= 0)
                    cfg.Layers[idx] = layer;
                else
                    cfg.Layers.Add(layer);
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
            foreach (var aq in BuildAquaticLayers())
                L.Add(aq);

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
        private const string GenericRoot = "Assets/Synty/PolygonGeneric";
        private const string CoreRoot = "Assets/Synty/PNB_Core";

        private static GameObject LoadCorePrefab(string name)
        {
            var list = FindPrefabs(CoreRoot, name);
            if (list.Count == 0)
                Debug.LogWarning($"[AniML] Core prefab not found: {name}");
            return list.Count > 0 ? list[0] : null;
        }
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
            // Migrate: assets serialized before the red-rock pass carry the
            // washed-out tan strata colors — punch them to actual red.
            if (s.CanyonRockLight == new Color(0.72f, 0.45f, 0.28f))
            {
                s.CanyonRockLight = new Color(0.80f, 0.42f, 0.22f);
                s.CanyonRockDark = new Color(0.58f, 0.26f, 0.15f);
                EditorUtility.SetDirty(s);
                AssetDatabase.SaveAssets();
            }
            return s;
        }

        // ---- Landform profiles ----

        [MenuItem("AniML/World/Profile/Temperate Isles")]
        public static void ProfileTemperate() => SetProfileAndGenerate(LandformProfile.TemperateIsles);

        [MenuItem("AniML/World/Profile/Alpine Granite")]
        public static void ProfileAlpine() => SetProfileAndGenerate(LandformProfile.AlpineGranite);

        [MenuItem("AniML/World/Profile/Canyonlands")]
        public static void ProfileCanyonlands() => SetProfileAndGenerate(LandformProfile.Canyonlands);

        [MenuItem("AniML/World/Profile/Volcano Isle")]
        public static void ProfileVolcano() => SetProfileAndGenerate(LandformProfile.VolcanoIsle);

        [MenuItem("AniML/World/Profile/Swamp Delta")]
        public static void ProfileSwampDelta() => SetProfileAndGenerate(LandformProfile.SwampDelta);

        /// <summary>Log every wonder and summit with its position, so you
        /// can fly straight to THE canyon instead of hunting for it.</summary>
        [MenuItem("AniML/World/Print Wonder Locations")]
        public static void PrintWonderLocations()
        {
            var world = Object.FindFirstObjectByType<ProceduralWorld>();
            if (world == null)
            {
                Debug.LogWarning("[AniML] No world in the scene.");
                return;
            }
            world.EnsureSamplers();
            foreach (var w in world.Wonders)
                Debug.Log($"[AniML] {w.Kind} at ({w.Pos.x:F0}, {w.Pos.z:F0})");
            foreach (var s in world.Summits)
                Debug.Log($"[AniML] Summit at ({s.Pos.x:F0}, {s.Pos.z:F0})");
            if (world.CanyonCrossing is Vector2 cc)
                Debug.Log($"[AniML] Canyon crossing at ({cc.x:F0}, {cc.y:F0})");
            foreach (var f in world.Fords)
                Debug.Log($"[AniML] Ford at ({f.Pos.x:F0}, {f.Pos.y:F0})");
        }

        private static void SetProfileAndGenerate(LandformProfile p)
        {
            var s = LoadOrCreateSettings();
            s.Profile = p;
            EditorUtility.SetDirty(s);
            AssetDatabase.SaveAssets();
            GenerateInScene();
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

        private const string PostProfilePath = "Assets/_Project/Settings/PostProfile.asset";
        private const string CloudCookiePath = "Assets/_Project/Art/Textures/T_CloudCookie.asset";

        /// <summary>Soft procedural cloud-shadow cookie for the sun.</summary>
        private static Texture2D EnsureCloudCookie()
        {
            var t = AssetDatabase.LoadAssetAtPath<Texture2D>(CloudCookiePath);
            if (t != null)
                return t;
            System.IO.Directory.CreateDirectory("Assets/_Project/Art/Textures");
            const int res = 256;
            t = new Texture2D(res, res, TextureFormat.RGBA32, true, true)
            {
                wrapMode = TextureWrapMode.Repeat,
                name = "T_CloudCookie",
            };
            for (int y = 0; y < res; y++)
            {
                for (int x = 0; x < res; x++)
                {
                    float n = 0f, amp = 1f, freq = 3f;
                    for (int o = 0; o < 3; o++)
                    {
                        // Tileable-ish: sample far from origin, mirror-free.
                        n += Mathf.PerlinNoise(x / (float)res * freq + 31.7f,
                            y / (float)res * freq + 57.3f) * amp;
                        amp *= 0.5f;
                        freq *= 2f;
                    }
                    n /= 1.75f;
                    float cloud = Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(0.52f, 0.78f, n));
                    float v = 1f - cloud * 0.42f;   // shadows, never black
                    t.SetPixel(x, y, new Color(v, v, v, 1f));
                }
            }
            t.Apply(true);
            AssetDatabase.CreateAsset(t, CloudCookiePath);
            AssetDatabase.SaveAssets();
            return t;
        }

        /// <summary>
        /// One global post-processing volume: bloom (lava, fireflies, and
        /// sun-glint actually glow), gentle grade, soft vignette. The art
        /// direction stays "stylized, not cartoonish" — all subtle.
        /// </summary>
        private static void EnsurePostProcessing()
        {
            var profile = AssetDatabase.LoadAssetAtPath<UnityEngine.Rendering.VolumeProfile>(PostProfilePath);
            if (profile == null)
            {
                profile = ScriptableObject.CreateInstance<UnityEngine.Rendering.VolumeProfile>();
                AssetDatabase.CreateAsset(profile, PostProfilePath);

                var bloom = profile.Add<UnityEngine.Rendering.Universal.Bloom>(false);
                bloom.intensity.Override(0.55f);
                bloom.threshold.Override(1.0f);
                bloom.scatter.Override(0.6f);

                var vignette = profile.Add<UnityEngine.Rendering.Universal.Vignette>(false);
                vignette.intensity.Override(0.22f);
                vignette.smoothness.Override(0.42f);

                var grade = profile.Add<UnityEngine.Rendering.Universal.ColorAdjustments>(false);
                grade.saturation.Override(9f);
                grade.contrast.Override(6f);

                AssetDatabase.SaveAssets();
            }

            var go = GameObject.Find("PostVolume");
            if (go == null)
                go = new GameObject("PostVolume");
            var vol = go.GetComponent<UnityEngine.Rendering.Volume>();
            if (vol == null)
                vol = go.AddComponent<UnityEngine.Rendering.Volume>();
            vol.isGlobal = true;
            vol.sharedProfile = profile;
        }

        /// <summary>
        /// Turn on the Synty foliage shader's per-material light wind so
        /// trees and grass sway (their demos do; the dials ship zeroed in
        /// several packs).
        /// </summary>
        private static void EnableFoliageWind()
        {
            var guids = AssetDatabase.FindAssets("t:Material", new[] { "Assets/Synty" });
            int touched = 0;
            foreach (var guid in guids)
            {
                var mat = AssetDatabase.LoadAssetAtPath<Material>(AssetDatabase.GUIDToAssetPath(guid));
                if (mat == null || mat.shader == null || !mat.shader.name.Contains("Foliage"))
                    continue;
                bool dirty = false;
                if (mat.HasProperty("_Enable_Light_Wind") && mat.GetFloat("_Enable_Light_Wind") < 0.5f)
                {
                    mat.SetFloat("_Enable_Light_Wind", 1f);
                    dirty = true;
                }
                if (mat.HasProperty("_Light_Wind_Strength") && mat.GetFloat("_Light_Wind_Strength") < 0.01f)
                {
                    mat.SetFloat("_Light_Wind_Strength", 0.45f);
                    dirty = true;
                }
                if (dirty)
                {
                    EditorUtility.SetDirty(mat);
                    touched++;
                }
            }
            if (touched > 0)
            {
                AssetDatabase.SaveAssets();
                Debug.Log($"[AniML] Enabled light wind on {touched} foliage materials.");
            }
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
            Wire("RedArch", AridRoot, "SM_Env_Rock_Arch_01");
            placer.SetMatSlot("RedArch", AssetDatabase.LoadAssetAtPath<Material>(RedRockMatPath));
            Wire("LavaPool", AridRoot, "SM_Env_Lava_Pool_01");
            Wire("LavaRiverA", AridRoot, "SM_Env_Lava_River_01");
            Wire("LavaRiverB", AridRoot, "SM_Env_Lava_River_02");
            Wire("LavaRiverC", AridRoot, "SM_Env_Lava_River_03");
            Wire("LavaRiverD", AridRoot, "SM_Env_Lava_River_04");
            Wire("LavaRiverE", AridRoot, "SM_Env_Lava_River_05");
            Wire("LavaSpreadA", AridRoot, "SM_Env_Lava_Spread_01");
            Wire("LavaEmbers", AridRoot, "FX_Lava_Ember_01");
            Wire("LavaSmoke", AridRoot, "FX_Smoke_Large_01");
            Wire("StalactiteA", GenericRoot, "SM_Gen_Env_Stalactite_01");
            Wire("StalactiteB", GenericRoot, "SM_Gen_Env_Stalactite_02");
            Wire("StalactiteC", GenericRoot, "SM_Gen_Env_Stalactite_03");
            Wire("StalactiteD", GenericRoot, "SM_Gen_Env_Stalactite_04");
            Wire("CaveDrip", AlpineRoot, "FX_Water_Drip_01");
            // The Alpine "Rock_Cliff" set IS glacier ice — its honest use.
            Wire("GlacierIceA", AlpineRoot, "SM_Env_Rock_Cliff_01");
            Wire("GlacierIceB", AlpineRoot, "SM_Env_Rock_Cliff_04");
            Wire("GlacierIceC", AlpineRoot, "SM_Env_Rock_Cliff_07");
            Wire("IceArch", AlpineRoot, "SM_Env_Rock_Cliff_Arch_01");
            Wire("Cave", TropicalRoot, "SM_Env_Rock_Cave_01");
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
            var stylized = Shader.Find("AniML/StylizedWater");
            var mat = AssetDatabase.LoadAssetAtPath<Material>(WaterMaterialPath);
            if (mat == null)
            {
                mat = new Material(stylized != null ? stylized : Shader.Find("Universal Render Pipeline/Lit"))
                    { name = "M_Water_Placeholder" };
                AssetDatabase.CreateAsset(mat, WaterMaterialPath);
                AssetDatabase.SaveAssets();
            }
            // Migrate pre-stylized assets onto the real water shader; the
            // open sea carries the visible swell.
            if (stylized != null && mat.shader != stylized)
            {
                mat.shader = stylized;
                EditorUtility.SetDirty(mat);
            }
            if (mat.HasProperty("_BobAmp") && !Mathf.Approximately(mat.GetFloat("_BobAmp"), 0.05f))
            {
                mat.SetFloat("_BobAmp", 0.05f);
                EditorUtility.SetDirty(mat);
                AssetDatabase.SaveAssets();
            }
            return mat;
        }

        private const string LakeMaterialPath = "Assets/_Project/Art/Materials/M_Water_Lake.mat";

        private static Material EnsureLakeWaterMaterial()
        {
            var stylized = Shader.Find("AniML/StylizedWater");
            if (stylized == null)
                return EnsureWaterMaterial();
            var mat = AssetDatabase.LoadAssetAtPath<Material>(LakeMaterialPath);
            if (mat == null)
            {
                mat = new Material(stylized) { name = "M_Water_Lake" };
                mat.SetFloat("_BobAmp", 0.012f);   // near-still: no ebbing shores
                mat.SetFloat("_DepthRange", 4f);
                mat.SetColor("_ShallowColor", new Color(0.34f, 0.62f, 0.6f, 0.6f));
                AssetDatabase.CreateAsset(mat, LakeMaterialPath);
                AssetDatabase.SaveAssets();
            }
            return mat;
        }

        private const string RiverMaterialPath = "Assets/_Project/Art/Materials/M_Water_River.mat";

        private static Material EnsureRiverWaterMaterial()
        {
            var stylized = Shader.Find("AniML/StylizedWater");
            if (stylized == null)
                return EnsureWaterMaterial();
            var mat = AssetDatabase.LoadAssetAtPath<Material>(RiverMaterialPath);
            if (mat == null)
            {
                mat = new Material(stylized) { name = "M_Water_River" };
                mat.SetFloat("_FlowUV", 1f);
                mat.SetColor("_ShallowColor", new Color(0.42f, 0.72f, 0.74f, 0.6f));
                mat.SetFloat("_DepthRange", 3f);
                mat.SetFloat("_FoamDepth", 0.85f);
                AssetDatabase.CreateAsset(mat, RiverMaterialPath);
                AssetDatabase.SaveAssets();
            }
            return mat;
        }

        /// <summary>
        /// Wire the ambient-effects slots: cascade FX and biome fauna. Every
        /// key matches a Species/Spawn call in AmbientFX.PlaceFauna.
        /// </summary>
        private static void WireAmbientFX(AmbientFX fx)
        {
            void Wire(string key, string packRoot, string prefabName)
            {
                var list = FindPrefabs(packRoot, prefabName);
                if (list.Count > 0)
                    fx.SetSlot(key, list[0]);
                else
                    Debug.LogWarning($"[AniML] AmbientFX prefab not found: {prefabName} under {packRoot}");
            }

            Wire("CascadeFoam", GenericRoot, "FX_Waterfall_Foam_01");
            Wire("CascadeImpact", TropicalRoot, "FX_Waterfall_Impact_01");
            Wire("Hawk", AlpineRoot, "FX_Hawk_01");
            Wire("Vulture", AridRoot, "FX_Vulture_01");
            Wire("Birds", TropicalRoot, "FX_Birds_01");
            Wire("Cranes", SwampRoot, "FX_Cranes_01");
            Wire("Frogs", SwampRoot, "FX_Frogs_01");
            Wire("Fireflies", SwampRoot, "FX_Fireflies_01");
            Wire("Butterflies", MeadowRoot, "FX_Butterflies_Monarch_01");
            Wire("Dragonflies", SwampRoot, "FX_Dragonflies_01");
            Wire("Crabs", TropicalRoot, "FX_Crabs_01");
            Wire("FishSmall", TropicalRoot, "FX_Fish_Small_01");
            Wire("FishMedium", TropicalRoot, "FX_Fish_Medium_01");
            Wire("Tumbleweed", AridRoot, "FX_TumbleWeed_01");
            Wire("DustBlowing", AridRoot, "FX_Dust_Blowing_01");
            Wire("Bats", SwampRoot, "FX_Bats_01");
            // World memory props (non-animal only — bones arrive when the
            // real ecosystem drops them at actual kills).
            Wire("MushroomA", MeadowRoot, "SM_Prop_Mushroom_01");
            Wire("MushroomB", MeadowRoot, "SM_Prop_Mushroom_02");
            Wire("MushroomC", MeadowRoot, "SM_Prop_Mushroom_03");
            Wire("VineA", TropicalRoot, "SM_Env_Vines_01");
            Wire("VineB", TropicalRoot, "SM_Env_Vines_02");
            Wire("BurnSnagA", AridRoot, "SM_Env_Tree_Dead_01");
            Wire("BurnSnagB", AridRoot, "SM_Env_Tree_Dead_02");
            Wire("Aurora", AlpineRoot, "FX_Aurora_Mesh_01");
            Wire("AutumnLeaves", MeadowRoot, "FX_Leaves_Blowing_Meadows_01");
            EditorUtility.SetDirty(fx);
        }
    }
}
