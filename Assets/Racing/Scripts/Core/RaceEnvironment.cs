using System.Collections.Generic;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>
    /// Owns the track and N agents (explicit index order). Physics runs in Script mode:
    /// PhysicsStep() = clear sensors → Physics.Simulate(dt) → AfterPhysicsStep for i = 0..N-1 (C0.8).
    /// </summary>
    [DefaultExecutionOrder(-900)]
    public sealed class RaceEnvironment : MonoBehaviour
    {
        [SerializeField] SimConfig simConfig;
        [SerializeField] VehicleConfig vehicleConfig;
        [SerializeField] TrackDefinition trackDefinition;
        [SerializeField] int numAgents = 1;
        [SerializeField] long seed = 1000;
        [SerializeField] StartMode startMode = StartMode.EvalGrid;
        [SerializeField] Material roadMaterial, wallMaterial, groundMaterial, carMaterial;

        readonly List<RaceAgentCore> _agents = new List<RaceAgentCore>();
        DeterministicRng[] _rngs;
        AgentStepResult[] _results;

        public bool IsInitialized { get; private set; }
        public TrackRuntime TrackRuntime { get; private set; }
        public TrackGeometry Track => TrackRuntime.Geometry;
        public IReadOnlyList<RaceAgentCore> Agents => _agents;
        public SimConfig Sim => simConfig;
        public VehicleConfig Vehicle => vehicleConfig;
        public TrackDefinition TrackDef => trackDefinition;
        public long StepIndex { get; private set; }
        public StartMode CurrentStartMode { get; private set; }
        public int MaxLaps { get; private set; }
        public long Seed { get; private set; }

        void Awake()
        {
            if (!IsInitialized && trackDefinition != null && vehicleConfig != null && simConfig != null)
                Initialize(trackDefinition, vehicleConfig, simConfig, numAgents, seed, startMode);
        }

        public static RaceEnvironment Create(TrackDefinition track, VehicleConfig vehicle, SimConfig sim, int agents, long seed, StartMode mode)
        {
            var go = new GameObject("RaceEnvironment");
            var env = go.AddComponent<RaceEnvironment>();
            env.Initialize(track, vehicle, sim, agents, seed, mode);
            return env;
        }

        public void Initialize(TrackDefinition track, VehicleConfig vehicle, SimConfig sim, int agents, long initialSeed, StartMode mode)
        {
            CultureBootstrap.Apply();
            trackDefinition = track;
            vehicleConfig = vehicle;
            simConfig = sim;
            numAgents = Mathf.Max(1, agents);

            Physics.simulationMode = SimulationMode.Script;
            Time.fixedDeltaTime = sim.fixedDeltaTime;
            Physics.IgnoreLayerCollision(RacingLayers.Car, RacingLayers.Car, true);

            var trackGo = new GameObject("Track");
            trackGo.transform.SetParent(transform, false);
            TrackRuntime = trackGo.AddComponent<TrackRuntime>();
            TrackRuntime.Build(track, roadMaterial, wallMaterial, groundMaterial);

            _agents.Clear();
            _rngs = new DeterministicRng[numAgents];
            _results = new AgentStepResult[numAgents];
            for (int i = 0; i < numAgents; i++)
            {
                VehicleController ctrl = VehicleFactory.Create(vehicle, sim, transform, "Car_" + i, carMaterial);
                var agent = ctrl.gameObject.AddComponent<RaceAgentCore>();
                agent.Initialize(i, TrackRuntime.Geometry, ctrl, sim, vehicle);
                _agents.Add(agent);
                _rngs[i] = new DeterministicRng(initialSeed + i);
            }
            IsInitialized = true;
            ResetAll(initialSeed, mode, 0);
        }

        /// <summary>Reseeds agent i with (seed + i) and respawns everyone. Mode and maxLaps stay until the next ResetAll.</summary>
        public void ResetAll(long newSeed, StartMode mode, int maxLaps)
        {
            Seed = newSeed;
            CurrentStartMode = mode;
            MaxLaps = maxLaps;
            for (int i = 0; i < _agents.Count; i++)
            {
                _rngs[i].Reseed(newSeed + i);
                ResetAgent(i);
            }
            Physics.SyncTransforms();
        }

        public void ResetAgent(int i)
        {
            SpawnSpec spawn = SpawnSampler.Sample(CurrentStartMode, _rngs[i], Track);
            _agents[i].BeginEpisode(spawn);
        }

        /// <summary>Actions must already be applied (RaceAgentCore.ApplyAction). Returns a reused array.</summary>
        public AgentStepResult[] PhysicsStep()
        {
            for (int i = 0; i < _agents.Count; i++) _agents[i].BeforePhysicsStep();
            Physics.Simulate(simConfig.fixedDeltaTime);
            for (int i = 0; i < _agents.Count; i++) _results[i] = _agents[i].AfterPhysicsStep();
            StepIndex++;
            return _results;
        }

        /// <summary>M1 part of env_config_hash (M2 appends RewardConfig, C0.11).</summary>
        public string ComputeEnvConfigHash(params IHashableConfig[] extra)
        {
            var parts = new List<IHashableConfig> { simConfig, vehicleConfig, trackDefinition, new ObsLayoutPart() };
            if (extra != null) parts.AddRange(extra);
            return ConfigHash.Compute(parts.ToArray());
        }

        sealed class ObsLayoutPart : IHashableConfig
        {
            public void AppendCanonical(SortedDictionary<string, string> kv) => kv["obs.layoutHash"] = ObservationSpec.LayoutHash;
        }
    }
}
