using System.IO;
using System.Linq;
using Racing.Core;
using Racing.MLAgents;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Racing.Editor
{
    /// <summary>M2 editor tooling: RewardConfig asset, Race_MLAgents scene, training player build, in-editor benchmark.</summary>
    public static class MlaTools
    {
        const string Root = "Assets/Racing";
        public const string RewardConfigPath = Root + "/Config/RewardConfig.asset";
        public const string MlaScenePath = Root + "/Scenes/Race_MLAgents.unity";
        public static string BuildPath => RepoPaths.MlaExe;
        public const string ModelsAssetDir = Root + "/Models";

        [MenuItem("Racing/Setup ML-Agents Scene (M2)")]
        public static void Setup()
        {
            if (AssetDatabase.LoadAssetAtPath<RewardConfig>(RewardConfigPath) == null)
                AssetDatabase.CreateAsset(ScriptableObject.CreateInstance<RewardConfig>(), RewardConfigPath);
            if (!AssetDatabase.IsValidFolder(ModelsAssetDir)) AssetDatabase.CreateFolder(Root, "Models");
            AssetDatabase.SaveAssets();

            // Sandbox also gets the RewardConfig asset reference (hash parity with the training scene).
            Scene sandbox = EditorSceneManager.OpenScene(ProjectSetup.SandboxScenePath, OpenSceneMode.Single);
            var sandboxEnv = Object.FindAnyObjectByType<RaceEnvironment>();
            if (sandboxEnv != null)
            {
                var so = new SerializedObject(sandboxEnv);
                so.FindProperty("rewardConfig").objectReferenceValue = AssetDatabase.LoadAssetAtPath<RewardConfig>(RewardConfigPath);
                so.ApplyModifiedPropertiesWithoutUndo();
                EditorSceneManager.SaveScene(sandbox);
            }

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var light = new GameObject("Directional Light").AddComponent<Light>();
            light.type = LightType.Directional;
            light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            var envGo = new GameObject("RaceEnvironment");
            var env = envGo.AddComponent<RaceEnvironment>();
            var eso = new SerializedObject(env);
            eso.FindProperty("simConfig").objectReferenceValue = AssetDatabase.LoadAssetAtPath<SimConfig>(ProjectSetup.SimConfigPath);
            eso.FindProperty("vehicleConfig").objectReferenceValue = AssetDatabase.LoadAssetAtPath<VehicleConfig>(ProjectSetup.VehicleConfigPath);
            eso.FindProperty("trackDefinition").objectReferenceValue = AssetDatabase.LoadAssetAtPath<TrackDefinition>(ProjectSetup.TrackDefinitionPath);
            eso.FindProperty("rewardConfig").objectReferenceValue = AssetDatabase.LoadAssetAtPath<RewardConfig>(RewardConfigPath);
            eso.FindProperty("initializeOnAwake").boolValue = false;
            foreach (string m in new[] { "road", "wall", "ground", "car" })
                eso.FindProperty(m + "Material").objectReferenceValue =
                    AssetDatabase.LoadAssetAtPath<Material>($"{Root}/Materials/{char.ToUpperInvariant(m[0]) + m.Substring(1)}.mat");
            eso.ApplyModifiedPropertiesWithoutUndo();

            var driverGo = new GameObject("MlaSimulationDriver");
            var driver = driverGo.AddComponent<MlaSimulationDriver>();
            var dso = new SerializedObject(driver);
            dso.FindProperty("environment").objectReferenceValue = env;
            dso.ApplyModifiedPropertiesWithoutUndo();

            var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
            var cam = camGo.AddComponent<Camera>();
            cam.farClipPlane = 2000f;
            var chase = camGo.AddComponent<ChaseCamera>();
            var cso = new SerializedObject(chase);
            cso.FindProperty("environment").objectReferenceValue = env;
            cso.ApplyModifiedPropertiesWithoutUndo();

            EditorSceneManager.SaveScene(scene, MlaScenePath);
            var scenes = EditorBuildSettings.scenes.Where(s => s.path != MlaScenePath).ToList();
            scenes.Add(new EditorBuildSettingsScene(MlaScenePath, true));
            EditorBuildSettings.scenes = scenes.ToArray();
            Debug.Log("[Race] M2 setup complete: " + MlaScenePath);
        }

        [MenuItem("Racing/Build ML-Agents Player (M2)")]
        public static BuildReport BuildPlayer()
        {
            PlayerSettings.runInBackground = true;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.defaultScreenWidth = 640;
            PlayerSettings.defaultScreenHeight = 360;
            PlayerSettings.resizableWindow = true;
            PlayerSettings.usePlayerLog = true;
            var options = new BuildPlayerOptions
            {
                scenes = new[] { MlaScenePath },
                locationPathName = BuildPath,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None
            };
            BuildReport report = BuildPipeline.BuildPlayer(options);
            Debug.Log($"[Race] Build {report.summary.result}: {report.summary.outputPath} " +
                      $"({report.summary.totalSize / (1024 * 1024)} MB, {report.summary.totalTime.TotalSeconds:F0}s, errors={report.summary.totalErrors})");
            return report;
        }

        /// <summary>
        /// Opens Race_MLAgents, adds a configured BenchmarkRunner (unsaved) and enters play mode. A relative outputPath is
        /// repo-relative (e.g. outputs/benchmarks/eval/mlagents_s1.json).
        /// </summary>
        public static void StartBenchmark(int trainSeed, string modelAssetPath, string outputPath, string buildId)
        {
            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(modelAssetPath);
            if (model == null) throw new FileNotFoundException("ModelAsset not found", modelAssetPath);
            // Re-use an already open (possibly dirty from a previous run) scene: switching would prompt to save.
            if (SceneManager.GetActiveScene().path != MlaScenePath)
                EditorSceneManager.OpenScene(MlaScenePath, OpenSceneMode.Single);
            var driver = Object.FindAnyObjectByType<MlaSimulationDriver>();
            var runner = driver.GetComponent<BenchmarkRunner>();
            if (runner == null) runner = driver.gameObject.AddComponent<BenchmarkRunner>();
            runner.Setup(model, trainSeed, RepoPaths.Resolve(outputPath), buildId);
            EditorApplication.isPlaying = true;
        }
    }
}
