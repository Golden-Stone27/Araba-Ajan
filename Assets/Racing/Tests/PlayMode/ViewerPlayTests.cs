using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using Racing.Core;
using Racing.Viewer;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace Racing.Tests
{
    /// <summary>
    /// UI1: the viewer scene swaps tracks in place (no scene reload), keeps exactly one environment, spawns the car on the
    /// start line, frees the replaced track's resources and leaves the frozen Track_A configuration untouched.
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
