using System.Collections.Generic;
using System.Linq;
using Racing.Core;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Racing.Editor
{
    /// <summary>Idempotent M1 project setup: layers, physics/player settings, config assets, materials, Sandbox scene.</summary>
    public static class ProjectSetup
    {
        const string Root = "Assets/Racing";
        public const string SandboxScenePath = Root + "/Scenes/Sandbox.unity";
        public const string SimConfigPath = Root + "/Config/SimConfig.asset";
        public const string VehicleConfigPath = Root + "/Config/VehicleConfig.asset";
        public const string TrackDefinitionPath = Root + "/Config/TrackDefinition_A.asset";

        [MenuItem("Racing/Setup Project (M1)")]
        public static void Run()
        {
            SetupLayers();
            SetupPhysics();
            PlayerSettings.runInBackground = true;
            QualitySettings.vSyncCount = 0;

            LoadOrCreate<SimConfig>(SimConfigPath);
            LoadOrCreate<VehicleConfig>(VehicleConfigPath);
            LoadOrCreate<TrackDefinition>(TrackDefinitionPath);
            LoadOrCreateMaterial("Road", new Color(0.18f, 0.18f, 0.2f));
            LoadOrCreateMaterial("Wall", new Color(0.8f, 0.15f, 0.15f));
            LoadOrCreateMaterial("Ground", new Color(0.25f, 0.45f, 0.22f));
            LoadOrCreateMaterial("Car", new Color(0.1f, 0.35f, 0.9f));
            AssetDatabase.SaveAssets();
            CreateSandboxScene();
            AssetDatabase.SaveAssets();
            Debug.Log("[Race] Project setup complete. obs_layout_hash=" + ObservationSpec.LayoutHash);
        }

        static void SetupLayers()
        {
            var tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
            SerializedProperty layers = tagManager.FindProperty("layers");
            layers.GetArrayElementAtIndex(RacingLayers.Car).stringValue = "Car";
            layers.GetArrayElementAtIndex(RacingLayers.Wall).stringValue = "Wall";
            layers.GetArrayElementAtIndex(RacingLayers.Road).stringValue = "Road";
            tagManager.ApplyModifiedPropertiesWithoutUndo();
        }

        static void SetupPhysics()
        {
            var dyn = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/DynamicsManager.asset")[0]);
            SetIfExists(dyn, "m_EnableEnhancedDeterminism", p => p.boolValue = true);
            SetIfExists(dyn, "m_SimulationMode", p => p.intValue = (int)SimulationMode.Script);
            SetIfExists(dyn, "m_AutoSyncTransforms", p => p.boolValue = false);
            dyn.ApplyModifiedPropertiesWithoutUndo();
            Physics.IgnoreLayerCollision(RacingLayers.Car, RacingLayers.Car, true);
            Time.fixedDeltaTime = 0.02f;
        }

        static void SetIfExists(SerializedObject so, string name, System.Action<SerializedProperty> set)
        {
            SerializedProperty p = so.FindProperty(name);
            if (p != null) set(p);
            else Debug.LogWarning("[Race] Physics setting not found: " + name);
        }

        static T LoadOrCreate<T>(string path) where T : ScriptableObject
        {
            var asset = AssetDatabase.LoadAssetAtPath<T>(path);
            if (asset != null) return asset;
            asset = ScriptableObject.CreateInstance<T>();
            AssetDatabase.CreateAsset(asset, path);
            return asset;
        }

        static Material LoadOrCreateMaterial(string name, Color color)
        {
            string path = $"{Root}/Materials/{name}.mat";
            var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (mat != null) return mat;
            mat = new Material(Shader.Find("Standard")) { color = color };
            AssetDatabase.CreateAsset(mat, path);
            return mat;
        }

        static void CreateSandboxScene()
        {
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            // Load assets AFTER NewScene: switching scenes unloads unreferenced assets loaded before it,
            // which silently turned the config references into {fileID: 0}.
            var sim = AssetDatabase.LoadAssetAtPath<SimConfig>(SimConfigPath);
            var vehicle = AssetDatabase.LoadAssetAtPath<VehicleConfig>(VehicleConfigPath);
            var track = AssetDatabase.LoadAssetAtPath<TrackDefinition>(TrackDefinitionPath);
            var mats = new Dictionary<string, Material>();
            foreach (string m in new[] { "Road", "Wall", "Ground", "Car" })
                mats[m] = AssetDatabase.LoadAssetAtPath<Material>($"{Root}/Materials/{m}.mat");

            var light = new GameObject("Directional Light").AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.1f;
            light.shadows = LightShadows.Soft;
            light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            var envGo = new GameObject("RaceEnvironment");
            var env = envGo.AddComponent<RaceEnvironment>();
            var so = new SerializedObject(env);
            so.FindProperty("simConfig").objectReferenceValue = sim;
            so.FindProperty("vehicleConfig").objectReferenceValue = vehicle;
            so.FindProperty("trackDefinition").objectReferenceValue = track;
            so.FindProperty("numAgents").intValue = 1;
            so.FindProperty("seed").longValue = 1000;
            so.FindProperty("startMode").intValue = (int)StartMode.EvalGrid;
            so.FindProperty("roadMaterial").objectReferenceValue = mats["Road"];
            so.FindProperty("wallMaterial").objectReferenceValue = mats["Wall"];
            so.FindProperty("groundMaterial").objectReferenceValue = mats["Ground"];
            so.FindProperty("carMaterial").objectReferenceValue = mats["Car"];
            so.ApplyModifiedPropertiesWithoutUndo();

            var driverGo = new GameObject("SimulationDriver");
            var driver = driverGo.AddComponent<SimulationDriver>();
            var dso = new SerializedObject(driver);
            dso.FindProperty("environment").objectReferenceValue = env;
            dso.ApplyModifiedPropertiesWithoutUndo();
            var hud = driverGo.AddComponent<RaceHud>();
            var hso = new SerializedObject(hud);
            hso.FindProperty("driver").objectReferenceValue = driver;
            hso.ApplyModifiedPropertiesWithoutUndo();

            var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
            var cam = camGo.AddComponent<Camera>();
            cam.farClipPlane = 2000f;
            camGo.AddComponent<AudioListener>();
            var chase = camGo.AddComponent<ChaseCamera>();
            var cso = new SerializedObject(chase);
            cso.FindProperty("environment").objectReferenceValue = env;
            cso.ApplyModifiedPropertiesWithoutUndo();
            camGo.transform.SetPositionAndRotation(new Vector3(0f, 25f, -40f), Quaternion.Euler(30f, 0f, 0f));

            EditorSceneManager.SaveScene(scene, SandboxScenePath);
            var scenes = EditorBuildSettings.scenes.Where(s => s.path != SandboxScenePath).ToList();
            scenes.Insert(0, new EditorBuildSettingsScene(SandboxScenePath, true));
            EditorBuildSettings.scenes = scenes.ToArray();
        }
    }
}
