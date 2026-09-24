using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using Racing.Core;
using Racing.Viewer;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;
using UnityEngine.UIElements;

namespace Racing.Tests
{
    /// <summary>
    /// UI1: the viewer scene swaps tracks in place (no scene reload), keeps exactly one environment, spawns the car on the
    /// start line, frees the replaced track's resources and leaves the frozen Track_A configuration untouched; the UI Toolkit
    /// track picker and HUD drive and reflect it.
    /// </summary>
    public class ViewerPlayTests
    {
        const string ScenePath = "Assets/Racing/Scenes/Race_Viewer.unity";
        static readonly string[] TrackMeshNames = { "RoadVisual", "RoadCollider", "WallRightVisual", "WallLeftVisual" };

        Scene _scene;
        ViewerController _viewer;

        [UnitySetUp]
        public IEnumerator SetUp()
        {
#if UNITY_EDITOR
            Assert.IsNotNull(UnityEditor.AssetDatabase.LoadAssetAtPath<UnityEditor.SceneAsset>(ScenePath),
                ScenePath + " missing: run Racing/Viewer/Create Viewer Scene (UI1)");
            AsyncOperation op = UnityEditor.SceneManagement.EditorSceneManager.LoadSceneAsyncInPlayMode(
                ScenePath, new LoadSceneParameters(LoadSceneMode.Additive));
            while (!op.isDone) yield return null;
            _scene = SceneManager.GetSceneByPath(ScenePath);
            foreach (GameObject root in _scene.GetRootGameObjects())
                if (_viewer == null) _viewer = root.GetComponentInChildren<ViewerController>();
            Assert.IsNotNull(_viewer, "ViewerController in " + ScenePath);
            for (int i = 0; i < 10 && _viewer.Car == null; i++) yield return null;
            Assert.IsNotNull(_viewer.Car, "initial track loaded in Start");
#else
            Assert.Ignore("viewer scene tests run in the Editor only");
            yield break;
#endif
        }

        [UnityTearDown]
        public IEnumerator TearDown()
        {
            _viewer = null;
            if (_scene.IsValid() && _scene.isLoaded)
            {
                AsyncOperation op = SceneManager.UnloadSceneAsync(_scene);
                while (op != null && !op.isDone) yield return null;
            }
            yield return Resources.UnloadUnusedAssets();
        }

        [UnityTest]
        public IEnumerator DefaultsTo_TrackA_WithAutopilot()
        {
            Assert.AreEqual(TrackCatalog.BenchmarkId, _viewer.TrackName);
            Assert.AreEqual(0, _viewer.TrackIndex);
            Assert.IsTrue(_viewer.Autopilot, "autopilot on by default");
            Assert.AreEqual("90240ee2b1a58b5b", _viewer.Environment.EnvConfigHash, "Track_A env_config_hash (C0.14)");
            yield break;
        }

        [UnityTest]
        public IEnumerator SwitchingTracks_RebuildsInPlace_WithoutSceneReload()
        {
            int sceneCount = SceneManager.sceneCount;
            var handle = _scene.handle;
            int loads = _viewer.LoadCount;
            var seen = new HashSet<RaceEnvironment>();
            foreach (string name in new[] { "Track_D", "proc:7", "Track_B", "Track_C", "Track_A" })
            {
                RaceEnvironment previous = _viewer.Environment;
                Assert.IsTrue(_viewer.LoadTrack(name, out string error), name + ": " + error);
                Assert.IsNull(error);
                yield return null; // Destroy of the replaced environment completes at the end of the frame

                Assert.IsTrue(previous == null, name + ": previous environment destroyed");
                RaceEnvironment env = _viewer.Environment;
                Assert.IsTrue(seen.Add(env), name + ": new environment instance");
                Assert.IsTrue(env.gameObject.scene == _scene, name + ": environment lives in the viewer scene");
                Assert.AreEqual(name, _viewer.TrackName);
                Assert.AreEqual(name, env.TrackDef.trackId);
                Assert.AreEqual(++loads, _viewer.LoadCount);
                Assert.AreEqual(1, CountActive<RaceEnvironment>(), name + ": one live environment");
                Assert.AreEqual(1, CountActive<TrackRuntime>(), name + ": one track");
                Assert.AreEqual(1, CountActive<RaceAgentCore>(), name + ": one car");
                AssertOnStartLine(name);
                Assert.AreEqual(sceneCount, SceneManager.sceneCount, "no scene loaded or unloaded");
                Assert.AreEqual(handle, _scene.handle, "same scene");
                Assert.IsTrue(_scene.isLoaded, "viewer scene still loaded");
            }
            Assert.AreEqual(0, _viewer.TrackIndex);
            Assert.AreEqual("90240ee2b1a58b5b", _viewer.Environment.EnvConfigHash, "Track_A env_config_hash after switching");
        }

        [UnityTest]
        public IEnumerator InvalidTrack_KeepsCurrentEnvironment()
        {
            RaceEnvironment env = _viewer.Environment;
            foreach (string name in new[] { "Track_Z", "proc:", "proc:-1", "proc:abc", "" })
            {
                Assert.IsFalse(_viewer.LoadTrack(name, out string error), name);
                Assert.IsFalse(string.IsNullOrEmpty(error), name + ": error message");
                yield return null;
                Assert.IsTrue(env != null && env == _viewer.Environment, name + ": environment kept");
                Assert.AreEqual(TrackCatalog.BenchmarkId, _viewer.TrackName);
            }
        }

        [UnityTest]
        public IEnumerator Switching_FreesReplacedTrackResources()
        {
            var generated = new List<TrackDefinition>();
            foreach (string name in new[] { "proc:1", "Track_D", "proc:2", "Track_C", "proc:3", "Track_A" })
            {
                Assert.IsTrue(_viewer.LoadTrack(name, out string error), name + ": " + error);
                if (_viewer.TrackIndex < 0) generated.Add(_viewer.Environment.TrackDef);
                yield return null;
            }
            yield return Resources.UnloadUnusedAssets();

            Assert.AreEqual(3, generated.Count);
            foreach (TrackDefinition def in generated)
                Assert.IsTrue(def == null, "procedural definition destroyed after switching away");
            var counts = new Dictionary<string, int>();
            foreach (Mesh m in Resources.FindObjectsOfTypeAll<Mesh>())
                if (System.Array.IndexOf(TrackMeshNames, m.name) >= 0)
                    counts[m.name] = counts.TryGetValue(m.name, out int c) ? c + 1 : 1;
            foreach (string mesh in new[] { "RoadVisual", "WallRightVisual", "WallLeftVisual" })
                Assert.AreEqual(1, counts.TryGetValue(mesh, out int c) ? c : 0, mesh + " meshes alive (Track_A only)");
            Assert.IsFalse(counts.ContainsKey("RoadCollider"), "Track_A has no MeshStrip collider");
        }

        [UnityTest]
        public IEnumerator ResetCar_PutsFreshCarOnStartLine()
        {
            RaceAgentCore car = _viewer.Car;
            float timeout = Time.time + 20f;
            while (car.Projection.S < 30f && Time.time < timeout) yield return null;
            Assert.GreaterOrEqual(car.Projection.S, 30f, "autopilot drives away from the start");
            Assert.Greater(_viewer.ElapsedSeconds, 0.5f);
            RaceEnvironment env = _viewer.Environment;

            _viewer.ResetCar();
            yield return null;
            Assert.IsTrue(car == null, "old car destroyed");
            Assert.IsTrue(env == _viewer.Environment, "same environment, same track");
            AssertOnStartLine("reset");
            Assert.Less(_viewer.ElapsedSeconds, 0.5f, "elapsed time restarts");
        }

        [UnityTest]
        public IEnumerator Ui_ListsCatalogTracksAndShowsLoadedTrack()
        {
            yield return null;
            VisualElement root = UiRoot();
            var dropdown = root.Q<DropdownField>(ViewerUI.TrackDropdownName);
            Assert.IsNotNull(dropdown, "track dropdown");
            CollectionAssert.AreEqual(new[] { "Track_A", "Track_B", "Track_C", "Track_D", ViewerUI.ProcChoice }, dropdown.choices);
            Assert.AreEqual("Track_A", dropdown.value);
            StringAssert.StartsWith("Track_A · Benchmark · L 1144.1 m · W 12.0 m · 114 kapı", root.Q<Label>(ViewerUI.InfoLabelName).text);
            Assert.AreEqual(DisplayStyle.None, root.Q<Label>(ViewerUI.ErrorLabelName).style.display.value, "no error");
            Assert.IsNotNull(root.Q<Label>(ViewerHud.SpeedLabelName), "HUD speed");
            Assert.IsNotNull(root.Q<Label>(ViewerHud.TimeLabelName), "HUD time");
        }

        [UnityTest]
        public IEnumerator Ui_Dropdown_LoadsCatalogAndProcTracks()
        {
            yield return null;
            VisualElement root = UiRoot();
            var dropdown = root.Q<DropdownField>(ViewerUI.TrackDropdownName);
            var seed = root.Q<TextField>(ViewerUI.SeedFieldName);
            var info = root.Q<Label>(ViewerUI.InfoLabelName);

            dropdown.value = "Track_C";
            yield return null;
            Assert.AreEqual("Track_C", _viewer.TrackName);
            StringAssert.StartsWith("Track_C · Speedway", info.text);
            Assert.AreEqual(DisplayStyle.None, seed.parent.style.display.value, "seed row hidden for catalog tracks");

            dropdown.value = ViewerUI.ProcChoice;
            yield return null;
            Assert.AreEqual("proc:1", _viewer.TrackName, "proc choice loads the seed field");
            Assert.AreEqual(DisplayStyle.Flex, seed.parent.style.display.value, "seed row shown");

            UiComponent<ViewerUI>().LoadSeed("42");
            yield return null;
            Assert.AreEqual("proc:42", _viewer.TrackName);
            Assert.AreEqual(-1, _viewer.TrackIndex);
            StringAssert.StartsWith("proc:42 · Procedural", info.text);
            Assert.AreEqual(ViewerUI.ProcChoice, dropdown.value);
            Assert.AreEqual("42", seed.value);
        }

        [UnityTest]
        public IEnumerator Ui_InvalidSeed_ShowsErrorAndKeepsTrack()
        {
            yield return null;
            VisualElement root = UiRoot();
            var error = root.Q<Label>(ViewerUI.ErrorLabelName);
            ViewerUI ui = UiComponent<ViewerUI>();
            RaceEnvironment env = _viewer.Environment;
            foreach (string digits in new[] { "", "12a", "99999999999999999999" })
            {
                ui.LoadSeed(digits);
                yield return null;
                Assert.AreEqual(DisplayStyle.Flex, error.style.display.value, "error shown for '" + digits + "'");
                Assert.IsTrue(env == _viewer.Environment, "track kept for '" + digits + "'");
            }
            ui.LoadSeed("3");
            yield return null;
            Assert.AreEqual(DisplayStyle.None, error.style.display.value, "error cleared by a successful load");
            Assert.AreEqual("proc:3", _viewer.TrackName);
        }

        [UnityTest]
        public IEnumerator Ui_AutopilotToggle_AndHud()
        {
            yield return null;
            VisualElement root = UiRoot();
            var speed = root.Q<Label>(ViewerHud.SpeedLabelName);
            var time = root.Q<Label>(ViewerHud.TimeLabelName);
            var mode = root.Q<Label>(ViewerHud.ModeLabelName);
            float timeout = Time.time + 10f;
            while (_viewer.Car.Telemetry.Speed < 5f && Time.time < timeout) yield return null;
            yield return null;
            Assert.AreEqual("OTOPİLOT", mode.text);
            Assert.Greater(int.Parse(speed.text, System.Globalization.CultureInfo.InvariantCulture), 10, "speed km/h");
            StringAssert.IsMatch(@"^Süre \d{2}:\d{2}\.\d{2}$", time.text);
            Assert.AreNotEqual("Süre 00:00.00", time.text);

            var toggle = root.Q<Toggle>();
            toggle.value = false;
            yield return null;
            Assert.IsFalse(_viewer.Autopilot, "toggle turns the autopilot off");
            Assert.AreEqual("KLAVYE", mode.text);
            _viewer.Autopilot = true; // P key path: the toggle follows the controller
            yield return null;
            Assert.IsTrue(toggle.value);
            Assert.AreEqual("OTOPİLOT", mode.text);
        }

        [Test]
        public void Hud_FormatsElapsedTime()
        {
            Assert.AreEqual("00:00.00", ViewerHud.FormatCentis(0));
            Assert.AreEqual("00:00.00", ViewerHud.FormatCentis(-5));
            Assert.AreEqual("01:23.45", ViewerHud.FormatCentis(8345));
            Assert.AreEqual("60:00.00", ViewerHud.FormatCentis(360000));
        }

        T UiComponent<T>() where T : Component
        {
            foreach (GameObject root in _scene.GetRootGameObjects())
            {
                var c = root.GetComponentInChildren<T>();
                if (c != null) return c;
            }
            Assert.Fail(typeof(T).Name + " in " + ScenePath);
            return null;
        }

        VisualElement UiRoot() => UiComponent<UIDocument>().rootVisualElement;

        void AssertOnStartLine(string label)
        {
            RaceAgentCore car = _viewer.Car;
            Assert.IsNotNull(car, label + ": car");
            Checkpoint start = _viewer.Environment.Track.GetCheckpoint(0);
            Vector3 d = car.transform.position - start.Position;
            d.y = 0f;
            Assert.Less(d.magnitude, 2f, label + ": car on the start line (EvalGrid, 1 m past it)");
            Assert.AreEqual(0, car.Telemetry.Laps, label + ": laps");
        }

        static int CountActive<T>() where T : Object => Object.FindObjectsByType<T>(FindObjectsInactive.Exclude).Length;
    }
}
