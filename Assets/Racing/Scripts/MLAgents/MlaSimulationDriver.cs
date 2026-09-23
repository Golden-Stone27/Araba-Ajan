using System;
using System.Collections.Generic;
using System.Globalization;
using Racing.Core;
using Unity.InferenceEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEngine;

namespace Racing.MLAgents
{
    /// <summary>
    /// M2 driver (C0.8): FixedUpdate = Academy.EnvironmentStep() → env.PhysicsStep() → AddReward /
    /// EndEpisode / EpisodeInterrupted. Physics stays in Script mode; the Academy never steps itself.
    /// Command line (player): -numAgents N, -envSeed S (default: derived from --mlagents-port).
    /// </summary>
    [DefaultExecutionOrder(-1000)]
    public sealed class MlaSimulationDriver : MonoBehaviour
    {
        [SerializeField] RaceEnvironment environment;
        [SerializeField] int numAgents = 16;
        [SerializeField] long envSeed = 1;
        [SerializeField] StartMode startMode = StartMode.TrainRandom;
        [SerializeField] int maxLaps;
        [Header("Inference (used when no trainer is connected)")]
        [SerializeField] ModelAsset model;
        [SerializeField] BehaviorType behaviorType = BehaviorType.Default;
        [SerializeField] bool deterministicInference = true;

        readonly List<RaceAgent> _agents = new List<RaceAgent>();
        StatsRecorder _stats;
        double[] _speedSum;
        int[] _steps;

        public RaceEnvironment Environment => environment;
        public IReadOnlyList<RaceAgent> Agents => _agents;
        public long StepCount { get; private set; }

        /// <summary>Invoked after every PhysicsStep, before agents are ended/reset (BenchmarkRunner hooks in here).</summary>
        public event Action<AgentStepResult[]> AfterPhysics;

        static readonly string[] ReasonKeys = BuildReasonKeys();

        static string[] BuildReasonKeys()
        {
            var k = new string[9];
            for (int r = 0; r < k.Length; r++) k[r] = "Race/TermReason/" + ((TermReason)r);
            return k;
        }

        /// <summary>Eval/benchmark configuration; must be called before Awake runs (i.e. on a disabled or not-yet-started scene object).</summary>
        public void Configure(int agents, long seed, StartMode mode, int laps, ModelAsset inferenceModel, BehaviorType type, bool deterministic)
        {
            numAgents = agents;
            envSeed = seed;
            startMode = mode;
            maxLaps = laps;
            model = inferenceModel;
            behaviorType = type;
            deterministicInference = deterministic;
        }

        void Awake()
        {
            CultureBootstrap.Apply();
            Academy.Instance.AutomaticSteppingEnabled = false;
            Physics.simulationMode = SimulationMode.Script;
            Application.runInBackground = true;

            var bench = GetComponent<BenchmarkRunner>();
            if (bench != null && bench.enabled) bench.ApplyTo(this);
            else ApplyCommandLine();

            if (environment == null) environment = FindAnyObjectByType<RaceEnvironment>();
            environment.InitializeFromSerialized(numAgents, envSeed, startMode);
            if (maxLaps > 0) environment.ResetAll(envSeed, startMode, maxLaps);

            int n = environment.Agents.Count;
            _speedSum = new double[n];
            _steps = new int[n];
            for (int i = 0; i < n; i++)
                _agents.Add(RaceAgent.Attach(environment.Agents[i], environment.Sim.decisionPeriod, model, behaviorType, deterministicInference));
            _stats = Academy.Instance.StatsRecorder;
            Debug.Log(string.Format(CultureInfo.InvariantCulture,
                "[Race] MLA driver: agents={0} seed={1} mode={2} maxLaps={3} env_config_hash={4} obs_layout_hash={5} trainer={6}",
                n, envSeed, startMode, maxLaps, environment.EnvConfigHash, ObservationSpec.LayoutHash, Academy.Instance.IsCommunicatorOn));
        }

        void ApplyCommandLine()
        {
            string[] args = System.Environment.GetCommandLineArgs();
            long portSeed = -1;
            for (int i = 0; i < args.Length - 1; i++)
            {
                string a = args[i];
                string v = args[i + 1];
                if (string.Equals(a, "-numAgents", StringComparison.Ordinal) && int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out int na))
                    numAgents = Mathf.Max(1, na);
                else if (string.Equals(a, "-envSeed", StringComparison.Ordinal) && long.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out long es))
                    envSeed = es;
                else if (string.Equals(a, "--mlagents-port", StringComparison.Ordinal) && long.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out long port))
                    portSeed = port;
            }
            // Parallel workers (--num-envs) get consecutive ports: give each its own spawn stream.
            if (portSeed >= 0 && !Array.Exists(args, x => string.Equals(x, "-envSeed", StringComparison.Ordinal)))
                envSeed = portSeed * 1000;
        }

        void FixedUpdate()
        {
            if (environment == null || !environment.IsInitialized) return;

            // Decision boundary: obs → policy; every step: OnActionReceived → core.ApplyAction (pre-physics).
            Academy.Instance.EnvironmentStep();
            AgentStepResult[] results = environment.PhysicsStep();
            StepCount++;
            AfterPhysics?.Invoke(results);

            for (int i = 0; i < results.Length; i++)
            {
                RaceAgent agent = _agents[i];
                RaceAgentCore core = agent.Core;
                AgentStepResult r = results[i];
                agent.AddReward(r.Reward);
                _speedSum[i] += core.State.IsFinite ? core.State.Velocity.magnitude : 0f;
                _steps[i]++;
                if (r.LapCompleted) _stats.Add("Race/LapTime", r.LapTime);

                if (!r.Terminated && !r.Truncated) continue;
                RecordEpisode(i, core, r.Reason);
                if (r.Terminated) agent.EndEpisode();
                else agent.EpisodeInterrupted();
            }
        }

        void RecordEpisode(int i, RaceAgentCore core, TermReason reason)
        {
            _stats.Add("Race/CompletionRate", TerminationPolicy.IsTruncation(reason) ? 1f : 0f);
            _stats.Add("Race/Laps", core.Tracker.LapsCompleted);
            _stats.Add("Race/Laps3Rate", core.Tracker.LapsCompleted >= 3 ? 1f : 0f);
            _stats.Add("Race/MeanSpeed", _steps[i] > 0 ? (float)(_speedSum[i] / _steps[i]) : 0f);
            for (int k = 1; k < ReasonKeys.Length; k++) _stats.Add(ReasonKeys[k], (int)reason == k ? 1f : 0f);
            RewardBreakdown rb = core.EpisodeRewards;
            _stats.Add("Race/Reward/Speed", rb.Speed);
            _stats.Add("Race/Reward/Wall", rb.Wall);
            _stats.Add("Race/Reward/Smooth", rb.Smooth);
            _stats.Add("Race/Reward/Checkpoint", rb.Checkpoint);
            _stats.Add("Race/Reward/Lap", rb.Lap);
            _stats.Add("Race/Reward/Terminal", rb.Terminal);
            _speedSum[i] = 0;
            _steps[i] = 0;
        }
    }
}
