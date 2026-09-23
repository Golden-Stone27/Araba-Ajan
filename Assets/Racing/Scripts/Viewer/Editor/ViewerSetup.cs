using Racing.Core;
using Racing.Viewer;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Racing.Editor
{
    /// <summary>
    /// UI1: builds Race_Viewer.unity (ViewerController + inactive RaceEnvironment template + chase camera).
    /// Built additively so the open scene is untouched; not added to the build settings (Editor play mode only).
    /// </summary>
    public static class ViewerSetup
    {
        const string Root = "Assets/Racing";
        public const string ViewerScenePath = Root + "/Scenes/Race_Viewer.unity";

        [MenuItem("Racing/Viewer/Create Viewer Scene (UI1)")]
        public static void CreateViewerScene()
        {
            if (EditorApplication.isPlaying)
            {
                Debug.LogError("[Viewer] exit play mode first");
                return;
            }
            if (SceneManager.GetSceneByPath(ViewerScenePath).isLoaded)
            {
                Debug.LogError("[Viewer] close " + ViewerScenePath + " first");
                return;
            }
            AssetDatabase.DeleteAsset(ViewerScenePath);
            Scene previous = SceneManager.GetActiveScene();
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
            SceneManager.SetActiveScene(scene);

            var light = new GameObject("Directional Light").AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.1f;
            light.shadows = LightShadows.Soft;
            light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            var envGo = new GameObject("RaceEnvironment (Template)");
            var env = envGo.AddComponent<RaceEnvironment>();
            var eso = new SerializedObject(env);
            eso.FindProperty("simConfig").objectReferenceValue = AssetDatabase.LoadAssetAtPath<SimConfig>(ProjectSetup.SimConfigPath);
            eso.FindProperty("vehicleConfig").objectReferenceValue = AssetDatabase.LoadAssetAtPath<VehicleConfig>(ProjectSetup.VehicleConfigPath);
            eso.FindProperty("trackDefinition").objectReferenceValue = AssetDatabase.LoadAssetAtPath<TrackDefinition>(ProjectSetup.TrackDefinitionPath);
            eso.FindProperty("rewardConfig").objectReferenceValue = AssetDatabase.LoadAssetAtPath<RewardConfig>(MlaTools.RewardConfigPath);
            eso.FindProperty("numAgents").intValue = 1;
            eso.FindProperty("initializeOnAwake").boolValue = false;
            foreach (string m in new[] { "road", "wall", "ground", "car" })
                eso.FindProperty(m + "Material").objectReferenceValue =
                    AssetDatabase.LoadAssetAtPath<Material>($"{Root}/Materials/{char.ToUpperInvariant(m[0]) + m.Substring(1)}.mat");
            eso.ApplyModifiedPropertiesWithoutUndo();
            envGo.SetActive(false);

            var viewer = new GameObject("Viewer").AddComponent<ViewerController>();
            var vso = new SerializedObject(viewer);
            vso.FindProperty("template").objectReferenceValue = env;
            vso.FindProperty("catalog").objectReferenceValue = AssetDatabase.LoadAssetAtPath<TrackCatalog>(TrackAssets.CatalogPath);
            vso.ApplyModifiedPropertiesWithoutUndo();

            var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
            camGo.AddComponent<Camera>().farClipPlane = 2000f;
            camGo.AddComponent<AudioListener>();
            var cso = new SerializedObject(camGo.AddComponent<ViewerChaseCamera>());
            cso.FindProperty("viewer").objectReferenceValue = viewer;
            cso.ApplyModifiedPropertiesWithoutUndo();
            camGo.transform.SetPositionAndRotation(new Vector3(0f, 25f, -40f), Quaternion.Euler(30f, 0f, 0f));

            EditorSceneManager.SaveScene(scene, ViewerScenePath);
            if (previous.IsValid() && previous.isLoaded) SceneManager.SetActiveScene(previous);
            EditorSceneManager.CloseScene(scene, true);
            Debug.Log("[Viewer] scene saved: " + ViewerScenePath);
        }
    }
}
