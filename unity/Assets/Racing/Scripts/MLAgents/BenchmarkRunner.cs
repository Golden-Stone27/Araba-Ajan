using System.Globalization;
using System.IO;
using Racing.Core;
using Unity.InferenceEngine;
using Unity.MLAgents.Policies;
using UnityEngine;

namespace Racing.MLAgents
{
    /// <summary>
    /// C0.10 in-process eval: N = 20 agents, EvalGrid (seed 1000 + i), maxLaps = 3, deterministic (μ) inference,
    /// time scale 20. Records each agent's first episode and writes a race-benchmark/v1 JSON (evaluator "unity-inproc").
    /// Lives next to MlaSimulationDriver; the driver calls ApplyTo in its Awake.
    /// </summary>
    [RequireComponent(typeof(MlaSimulationDriver))]
    public sealed class BenchmarkRunner : MonoBehaviour
    {
        public const int Episodes = 20;
        public const long SeedBase = 1000;
        public const int Laps = 3;

        [SerializeField] ModelAsset model;
        [SerializeField] int trainSeed = 1;
        [SerializeField] string outputPath = "../outputs/benchmarks/eval/mlagents_s1.json";  // relative to the Unity project
        [SerializeField] string buildId = "editor";
        [SerializeField] float timeScale = 20f;
        [SerializeField] bool exitWhenDone = true;

        MlaSimulationDriver _driver;
        BenchmarkRecorder _recorder;
        float _startRealtime;
        float _prevMaxDelta = -1f;

        public bool Done { get; private set; }
        public string OutputPath => outputPath;

        public void Setup(ModelAsset m, int seed, string output, string build)
        {
            model = m;
            trainSeed = seed;
            outputPath = output;
            buildId = build;
        }

        public void ApplyTo(MlaSimulationDriver driver)
        {
            _driver = driver;
            driver.Configure(Episodes, SeedBase, StartMode.EvalGrid, Laps, model, BehaviorType.InferenceOnly, true);
        }

        void Start()
        {
            if (_driver == null) _driver = GetComponent<MlaSimulationDriver>();
            _recorder = new BenchmarkRecorder(_driver.Environment, Laps);
            _driver.AfterPhysics += OnAfterPhysics;
            Time.timeScale = timeScale;
            _prevMaxDelta = Time.maximumDeltaTime;
            Time.maximumDeltaTime = 1f; // let FixedUpdate catch up at high time scale
            _startRealtime = Time.realtimeSinceStartup;
            Debug.Log("[Race] Benchmark started: model=" + (model != null ? model.name : "<none>") + " seed=" + trainSeed);
        }

        void OnAfterPhysics(AgentStepResult[] results)
        {
            if (Done) return;
            _recorder.Observe(results);
            if (!_recorder.AllDone) return;

            Done = true;
            string json = _recorder.ReportJson("mlagents-ppo", "unity-inproc", trainSeed, buildId);
            _recorder.Write(Path.GetFullPath(outputPath), json);
            Debug.Log(string.Format(CultureInfo.InvariantCulture, "[Race] Benchmark done in {0:F1}s real ({1} steps) → {2}\n{3}",
                Time.realtimeSinceStartup - _startRealtime, _driver.StepCount, outputPath, json));
            if (!exitWhenDone) return;
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }

        void OnDestroy()
        {
            if (_driver != null) _driver.AfterPhysics -= OnAfterPhysics;
            Time.timeScale = 1f;
            if (_prevMaxDelta > 0f) Time.maximumDeltaTime = _prevMaxDelta;
        }
    }
}
