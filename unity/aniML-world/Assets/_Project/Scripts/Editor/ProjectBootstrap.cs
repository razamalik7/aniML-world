using System.IO;
using AniML.CameraRig;
using AniML.Creatures;
using AniML.Sandbox;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace AniML.EditorTools
{
    /// <summary>
    /// One-shot project setup, safe to re-run: creates the URP pipeline assets,
    /// assigns them to Graphics/Quality settings, and builds the Sandbox scene.
    /// Run from the menu (AniML/Bootstrap Project) or batchmode -executeMethod.
    /// </summary>
    public static class ProjectBootstrap
    {
        private const string SettingsDir = "Assets/_Project/Settings";
        private const string ScenesDir = "Assets/_Project/Scenes";
        private const string ArtDir = "Assets/_Project/Art/Materials";
        private const string CreaturesDir = "Assets/_Project/Creatures";
        private const string ScenePath = ScenesDir + "/Sandbox.unity";

        private const string PrefabsDir = "Assets/_Project/Prefabs";
        private const string MalbersDeerFbxPath = "Assets/Malbers Animations/Animals Packs/01 Forest Pack/Deer/Models/Poly Art/Deer.FBX";
        private const string MalbersDeerSkinMatPath = "Assets/Malbers Animations/Animals Packs/01 Forest Pack/Deer/Materials/Poly Art/Deer Skin Common.mat";
        private const string MalbersDeerAntlersMatPath = "Assets/Malbers Animations/Animals Packs/01 Forest Pack/Deer/Materials/Poly Art/Antlers.mat";
        private const string DeerPrefabPath = PrefabsDir + "/Deer_PolyArt.prefab";

        [MenuItem("AniML/Bootstrap Project")]
        public static void Run()
        {
            EnsureFolders();
            var rpAsset = SetupUrp();
            var deer = EnsureDeerDefinition();
            BuildSandboxScene(deer);
            AssetDatabase.SaveAssets();
            Debug.Log($"[AniML] Bootstrap complete. URP={(rpAsset != null)} scene={ScenePath}");
        }

        private static void EnsureFolders()
        {
            foreach (var dir in new[] { SettingsDir, ScenesDir, ArtDir, CreaturesDir, PrefabsDir })
            {
                if (!Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }
            }
            AssetDatabase.Refresh();
        }

        private static UniversalRenderPipelineAsset SetupUrp()
        {
            var rpPath = SettingsDir + "/AniML_URP.asset";
            var rpAsset = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>(rpPath);
            if (rpAsset == null)
            {
                var rendererData = ScriptableObject.CreateInstance<UniversalRendererData>();
                AssetDatabase.CreateAsset(rendererData, SettingsDir + "/AniML_URP_Renderer.asset");
                rpAsset = UniversalRenderPipelineAsset.Create(rendererData);
                AssetDatabase.CreateAsset(rpAsset, rpPath);
            }

            rpAsset.shadowDistance = 150f;
            EditorUtility.SetDirty(rpAsset);

            GraphicsSettings.defaultRenderPipeline = rpAsset;

            int initialLevel = QualitySettings.GetQualityLevel();
            for (int i = 0; i < QualitySettings.names.Length; i++)
            {
                QualitySettings.SetQualityLevel(i, false);
                QualitySettings.renderPipeline = rpAsset;
            }
            QualitySettings.SetQualityLevel(initialLevel, false);
            return rpAsset;
        }

        public static CreatureDefinition EnsureDeerDefinition()
        {
            EnsureFolders();
            var path = CreaturesDir + "/Deer.asset";
            var def = AssetDatabase.LoadAssetAtPath<CreatureDefinition>(path);
            if (def == null)
            {
                def = ScriptableObject.CreateInstance<CreatureDefinition>();
                def.SpeciesId = "deer";
                def.DisplayName = "PolyArt Deer";
                def.MaxSpeed = 4.2f;
                def.TurnRate = 220f;
                AssetDatabase.CreateAsset(def, path);
            }

            var prefab = EnsurePolyArtDeerPrefab(def);
            if (prefab != null && def.Prefab != prefab)
            {
                def.Prefab = prefab;
                EditorUtility.SetDirty(def);
                AssetDatabase.SaveAssets();
            }

            return def;
        }

        public static GameObject EnsurePolyArtDeerPrefab(CreatureDefinition def)
        {
            var fbxModel = AssetDatabase.LoadAssetAtPath<GameObject>(MalbersDeerFbxPath);
            if (fbxModel == null)
            {
                Debug.LogWarning($"[AniML] Malbers PolyArt Deer FBX not found at '{MalbersDeerFbxPath}'; using procedural articulated skeleton fallback.");
                return AssetDatabase.LoadAssetAtPath<GameObject>(DeerPrefabPath);
            }

            var skinMat = AssetDatabase.LoadAssetAtPath<Material>(MalbersDeerSkinMatPath);
            var antlersMat = AssetDatabase.LoadAssetAtPath<Material>(MalbersDeerAntlersMatPath);

            var instance = (GameObject)PrefabUtility.InstantiatePrefab(fbxModel);
            instance.name = "Deer_PolyArt";
            instance.transform.localScale = Vector3.one * 0.7f;

            foreach (var anim in instance.GetComponentsInChildren<Animator>(true))
                anim.enabled = false;

            foreach (var smr in instance.GetComponentsInChildren<SkinnedMeshRenderer>(true))
            {
                smr.updateWhenOffscreen = true;
                string n = smr.gameObject.name;
                if (n.StartsWith("Deer") && skinMat != null)
                {
                    smr.sharedMaterial = skinMat;
                }
                else if (n.StartsWith("Antlers"))
                {
                    if (antlersMat != null)
                        smr.sharedMaterial = antlersMat;
                    smr.gameObject.SetActive(n == "Antlers 01");
                }
                else if (n.StartsWith("Magic") || n == "Spots")
                {
                    smr.gameObject.SetActive(false);
                }
            }

            if (instance.GetComponent<DeerRigDriver>() == null)
                instance.AddComponent<DeerRigDriver>();
            if (instance.GetComponent<DeerPolicyLocomotionMotor>() == null)
                instance.AddComponent<DeerPolicyLocomotionMotor>();
            if (instance.GetComponent<DeerEcologyBrain>() == null)
                instance.AddComponent<DeerEcologyBrain>();

            var ctrl = instance.GetComponent<CreatureController>();
            if (ctrl == null)
                ctrl = instance.AddComponent<CreatureController>();
            if (def != null)
            {
                var so = new SerializedObject(ctrl);
                so.FindProperty("definition").objectReferenceValue = def;
                so.ApplyModifiedPropertiesWithoutUndo();
            }

            var savedPrefab = PrefabUtility.SaveAsPrefabAsset(instance, DeerPrefabPath);
            Object.DestroyImmediate(instance);
            return savedPrefab;
        }

        private static void BuildSandboxScene(CreatureDefinition deer)
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

            // Camera → god camera rig
            var cam = Camera.main;
            if (cam != null && cam.GetComponent<GodCamera>() == null)
            {
                cam.gameObject.AddComponent<GodCamera>();
            }

            // Light: warm key light at a pleasant angle
            var light = Object.FindFirstObjectByType<Light>();
            if (light != null)
            {
                light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
                light.color = new Color(1f, 0.956f, 0.839f);
                light.shadows = LightShadows.Soft;
            }

            // Ground: 400 x 400 m plane with a flat stylized green
            var groundMat = new Material(Shader.Find("Universal Render Pipeline/Lit"))
            {
                name = "M_Ground_Placeholder",
            };
            groundMat.SetColor("_BaseColor", new Color(0.478f, 0.671f, 0.365f));
            groundMat.SetFloat("_Smoothness", 0f);
            AssetDatabase.CreateAsset(groundMat, ArtDir + "/M_Ground_Placeholder.mat");

            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.localScale = new Vector3(40f, 1f, 40f);
            ground.GetComponent<MeshRenderer>().sharedMaterial = groundMat;

            // Spawner with the deer placeholder definition
            var spawnerGo = new GameObject("SandboxSpawner");
            var spawner = spawnerGo.AddComponent<SandboxSpawner>();
            var so = new SerializedObject(spawner);
            so.FindProperty("definition").objectReferenceValue = deer;
            so.ApplyModifiedPropertiesWithoutUndo();

            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        }
    }
}
