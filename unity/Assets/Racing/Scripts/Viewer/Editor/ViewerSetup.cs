using Racing.Core;
using Racing.Viewer;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;

namespace Racing.Editor
{
    /// <summary>
    /// UI1: builds Race_Viewer.unity (ViewerController + inactive RaceEnvironment template + chase camera + UI Toolkit
    /// track picker and HUD; panel settings in Assets/Racing/UI).
    /// Built additively so other open scenes are untouched; not added to the build settings (Editor play mode only).
    /// </summary>
    public static class ViewerSetup
    {
        const string Root = "Assets/Racing";
        public const string ViewerScenePath = Root + "/Scenes/Race_Viewer.unity";
        public const string PanelSettingsPath = Root + "/UI/ViewerPanelSettings.asset";
        public const string ThemePath = Root + "/UI/ViewerTheme.tss";
        public const string StyleSheetPath = Root + "/UI/Viewer.uss";

        [MenuItem("Racing/Viewer/Create Viewer Scene (UI1)")]
        public static void CreateViewerScene()
        {
            if (EditorApplication.isPlaying)
            {
                Debug.LogError("[Viewer] exit play mode first");
                return;
            }
            // An open viewer scene without unsaved changes is rebuilt in place: as the only scene (or instead of a lone empty
            // untitled one) it is replaced and stays open; next to other scenes it is closed and reopened.
            // The file is overwritten, so its GUID stays.
            Scene open = SceneManager.GetSceneByPath(ViewerScenePath);
            bool reopen = open.isLoaded;
            if (reopen && open.isDirty)
            {
                Debug.LogError("[Viewer] save or discard the changes in " + ViewerScenePath + " first");
                return;
            }
            Scene active = SceneManager.GetActiveScene();
            bool emptyUntitled = string.IsNullOrEmpty(active.path) && !active.isDirty; // blocks additive scene creation
            bool single = SceneManager.sceneCount == 1 && (reopen || emptyUntitled);
            if (reopen && !single) EditorSceneManager.CloseScene(open, true);
            Scene previous = SceneManager.GetActiveScene();
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, single ? NewSceneMode.Single : NewSceneMode.Additive);
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

            var uiGo = new GameObject("ViewerUI");
            uiGo.AddComponent<UIDocument>().panelSettings = LoadOrCreatePanelSettings();
            var styleSheet = AssetDatabase.LoadAssetAtPath<StyleSheet>(StyleSheetPath);
            if (styleSheet == null) Debug.LogError("[Viewer] missing " + StyleSheetPath);
            foreach (MonoBehaviour ui in new MonoBehaviour[] { uiGo.AddComponent<ViewerUI>(), uiGo.AddComponent<ViewerHud>() })
            {
                var uso = new SerializedObject(ui);
                uso.FindProperty("viewer").objectReferenceValue = viewer;
                uso.FindProperty("styleSheet").objectReferenceValue = styleSheet;
                uso.ApplyModifiedPropertiesWithoutUndo();
            }

            EditorSceneManager.SaveScene(scene, ViewerScenePath);
            if (!single)
            {
                if (previous.IsValid() && previous.isLoaded) SceneManager.SetActiveScene(previous);
                EditorSceneManager.CloseScene(scene, true);
                if (reopen) EditorSceneManager.OpenScene(ViewerScenePath, OpenSceneMode.Additive);
            }
            Debug.Log("[Viewer] scene saved: " + ViewerScenePath);
        }

        /// <summary>Screen-space panel scaled from 1600×900 with the default runtime theme; settings are re-applied on every run.</summary>
        static PanelSettings LoadOrCreatePanelSettings()
        {
            var settings = AssetDatabase.LoadAssetAtPath<PanelSettings>(PanelSettingsPath);
            bool create = settings == null;
            if (create) settings = ScriptableObject.CreateInstance<PanelSettings>();
            settings.themeStyleSheet = AssetDatabase.LoadAssetAtPath<ThemeStyleSheet>(ThemePath);
            if (settings.themeStyleSheet == null) Debug.LogError("[Viewer] missing " + ThemePath);
            settings.scaleMode = PanelScaleMode.ScaleWithScreenSize;
            settings.referenceResolution = new Vector2Int(1600, 900);
            settings.screenMatchMode = PanelScreenMatchMode.MatchWidthOrHeight;
            settings.match = 0.5f;
            if (create) AssetDatabase.CreateAsset(settings, PanelSettingsPath);
            else EditorUtility.SetDirty(settings);
            AssetDatabase.SaveAssets();
            return settings;
        }
    }
}
