using System.Linq;
using Racing.Bridge;
using Racing.Core;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Racing.Editor
{
    /// <summary>
    /// M3 tooling: Race_Bridge scene (RaceEnvironment + BridgeDriver) and the bridge player build.
    /// Batch: Unity.exe -batchmode -quit -projectPath D:\RaceAgent -executeMethod Racing.Editor.BuildScript.BuildBridge
    /// </summary>
    public static class BuildScript
    {
        const string Root = "Assets/Racing";
        public const string BridgeScenePath = Root + "/Scenes/Race_Bridge.unity";
        public const string WatchScenePath = Root + "/Scenes/Race_Watch.unity";
        public const string BridgeBuildPath = "Builds/RaceEnv/RaceEnv.exe";

        [MenuItem("Racing/Setup Bridge Scene (M3)")]
        public static void SetupBridgeScene()
        {
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var light = new GameObject("Directional Light").AddComponent<Light>();
            light.type = LightType.Directional;
            light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            var env = new GameObject("RaceEnvironment").AddComponent<RaceEnvironment>();
            var eso = new SerializedObject(env);
            eso.FindProperty("simConfig").objectReferenceValue = AssetDatabase.LoadAssetAtPath<SimConfig>(ProjectSetup.SimConfigPath);
            eso.FindProperty("vehicleConfig").objectReferenceValue = AssetDatabase.LoadAssetAtPath<VehicleConfig>(ProjectSetup.VehicleConfigPath);
            eso.FindProperty("trackDefinition").objectReferenceValue = AssetDatabase.LoadAssetAtPath<TrackDefinition>(ProjectSetup.TrackDefinitionPath);
            eso.FindProperty("rewardConfig").objectReferenceValue = AssetDatabase.LoadAssetAtPath<RewardConfig>(MlaTools.RewardConfigPath);
            eso.FindProperty("initializeOnAwake").boolValue = false;
            foreach (string m in new[] { "road", "wall", "ground", "car" })
                eso.FindProperty(m + "Material").objectReferenceValue =
                    AssetDatabase.LoadAssetAtPath<Material>($"{Root}/Materials/{char.ToUpperInvariant(m[0]) + m.Substring(1)}.mat");
            eso.ApplyModifiedPropertiesWithoutUndo();

            var driver = new GameObject("BridgeDriver").AddComponent<BridgeDriver>();
            var dso = new SerializedObject(driver);
            dso.FindProperty("environment").objectReferenceValue = env;
            dso.ApplyModifiedPropertiesWithoutUndo();
            BindTrackCatalog(scene);

            var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
            camGo.AddComponent<Camera>().farClipPlane = 2000f;
            var cso = new SerializedObject(camGo.AddComponent<ChaseCamera>());
            cso.FindProperty("environment").objectReferenceValue = env;
            cso.ApplyModifiedPropertiesWithoutUndo();

            EditorSceneManager.SaveScene(scene, BridgeScenePath);
            var scenes = EditorBuildSettings.scenes.Where(s => s.path != BridgeScenePath).ToList();
            scenes.Add(new EditorBuildSettingsScene(BridgeScenePath, true));
            EditorBuildSettings.scenes = scenes.ToArray();
            Debug.Log("[Race] M3 bridge scene saved: " + BridgeScenePath);
        }

        /// <summary>
        /// Race_Bridge copy with a single agent for watching a policy live (python -m racing_rl.train.watch).
        /// Built additively so the open scene is untouched; not added to the build settings.
        /// </summary>
        [MenuItem("Racing/Setup Watch Scene (1 agent)")]
        public static void SetupWatchScene()
        {
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(BridgeScenePath) == null) SetupBridgeScene();
            AssetDatabase.DeleteAsset(WatchScenePath);
            if (!AssetDatabase.CopyAsset(BridgeScenePath, WatchScenePath))
            {
                Debug.LogError("[Race] cannot copy " + BridgeScenePath + " to " + WatchScenePath);
                return;
            }
            Scene scene = EditorSceneManager.OpenScene(WatchScenePath, OpenSceneMode.Additive);
            foreach (GameObject root in scene.GetRootGameObjects())
            foreach (BridgeDriver driver in root.GetComponentsInChildren<BridgeDriver>(true))
            {
                var dso = new SerializedObject(driver);
                dso.FindProperty("numAgents").intValue = 1;
                dso.ApplyModifiedPropertiesWithoutUndo();
            }
            BindTrackCatalog(scene);
            EditorSceneManager.SaveScene(scene);
            EditorSceneManager.CloseScene(scene, true);
            Debug.Log("[Race] watch scene saved: " + WatchScenePath);
        }

        /// <summary>
        /// M6: binds TrackCatalog.asset to the BridgeDriver of the existing bridge and watch scenes without rebuilding them
        /// (-trackName / -trackIndex need it). Scenes that are not open are opened additively and closed again.
        /// </summary>
        [MenuItem("Racing/Tracks/Bind Track Catalog To Bridge Scenes (M6)")]
        public static void BindTrackCatalogToBridgeScenes()
        {
            foreach (string path in new[] { BridgeScenePath, WatchScenePath })
            {
                if (AssetDatabase.LoadAssetAtPath<SceneAsset>(path) == null) continue;
                Scene scene = SceneManager.GetSceneByPath(path);
                bool wasOpen = scene.IsValid() && scene.isLoaded;
                if (!wasOpen) scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Additive);
                int n = BindTrackCatalog(scene);
                EditorSceneManager.SaveScene(scene);
                if (!wasOpen) EditorSceneManager.CloseScene(scene, true);
                Debug.Log($"[Race] track catalog bound to {n} BridgeDriver(s) in {path}");
            }
        }

        /// <summary>Sets BridgeDriver.catalog on every driver in the scene; returns how many were bound.</summary>
        static int BindTrackCatalog(Scene scene)
        {
            var catalog = AssetDatabase.LoadAssetAtPath<TrackCatalog>(TrackAssets.CatalogPath);
            if (catalog == null)
            {
                Debug.LogError("[Race] " + TrackAssets.CatalogPath + " not found (Racing/Tracks/Setup Track Catalog (M6))");
                return 0;
            }
            int n = 0;
            foreach (GameObject root in scene.GetRootGameObjects())
            foreach (BridgeDriver driver in root.GetComponentsInChildren<BridgeDriver>(true))
            {
                var dso = new SerializedObject(driver);
                dso.FindProperty("catalog").objectReferenceValue = catalog;
                dso.ApplyModifiedPropertiesWithoutUndo();
                n++;
            }
            return n;
        }

        [MenuItem("Racing/Build Bridge Player (M3)")]
        public static void BuildBridge()
        {
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(BridgeScenePath) == null) SetupBridgeScene();
            PlayerSettings.runInBackground = true;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.defaultScreenWidth = 640;
            PlayerSettings.defaultScreenHeight = 360;
            PlayerSettings.resizableWindow = true;
            PlayerSettings.usePlayerLog = true;
            BuildReport report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { BridgeScenePath },
                locationPathName = BridgeBuildPath,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None
            });
            Debug.Log($"[Race] Bridge build {report.summary.result}: {report.summary.outputPath} " +
                      $"({report.summary.totalSize / (1024 * 1024)} MB, {report.summary.totalTime.TotalSeconds:F0}s, errors={report.summary.totalErrors})");
            if (Application.isBatchMode && report.summary.result != BuildResult.Succeeded) EditorApplication.Exit(1);
        }
    }
}
