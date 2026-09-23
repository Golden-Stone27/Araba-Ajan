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
        [SerializeField] RewardConfig rewardConfig;
        [SerializeField] int numAgents = 1;
        [SerializeField] long seed = 1000;
        [SerializeField] StartMode startMode = StartMode.EvalGrid;
        [SerializeField] Material roadMaterial, wallMaterial, groundMaterial, carMaterial;
        [Tooltip("Off when a driver (e.g. MlaSimulationDriver) calls InitializeFromSerialized with runtime arguments.")]
        [SerializeField] bool initializeOnAwake = true;

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
        public RewardConfig Reward => rewardConfig;
        public long StepIndex { get; private set; }
        public StartMode CurrentStartMode { get; private set; }
        public int MaxLaps { get; private set; }
        public long Seed { get; private set; }

        void Awake()
        {
            if (initializeOnAwake && !IsInitialized && trackDefinition != null && vehicleConfig != null && simConfig != null)
                Initialize(trackDefinition, vehicleConfig, simConfig, numAgents, seed, startMode, rewardConfig);
        }

        /// <summary>
        /// Initializes with the serialized configs/materials but runtime agent count, seed and start mode.
        /// trackOverride (M6 bridge -trackName) replaces the serialized track; null keeps it.
        /// </summary>
        public void InitializeFromSerialized(int agents, long initialSeed, StartMode mode, TrackDefinition trackOverride = null) =>
            Initialize(trackOverride != null ? trackOverride : trackDefinition, vehicleConfig, simConfig, agents, initialSeed, mode, rewardConfig);

        public static RaceEnvironment Create(TrackDefinition track, VehicleConfig vehicle, SimConfig sim, int agents, long seed, StartMode mode,
                                             RewardConfig reward = null)
        {
            var go = new GameObject("RaceEnvironment");
            var env = go.AddComponent<RaceEnvironment>();
            env.Initialize(track, vehicle, sim, agents, seed, mode, reward);
            return env;
        }

        public void Initialize(TrackDefinition track, VehicleConfig vehicle, SimConfig sim, int agents, long initialSeed, StartMode mode,
                               RewardConfig reward = null)
        {
            CultureBootstrap.Apply();
            rewardConfig = reward != null ? reward : (rewardConfig != null ? rewardConfig : RewardConfig.CreateDefault());
            _envConfigHash = null;
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

            _rngs = new DeterministicRng[numAgents];
            _results = new AgentStepResult[numAgents];
            _initialSeed = initialSeed;
            _initialMode = mode;
            CreateAgents(initialSeed);
            IsInitialized = true;
            ResetAll(initialSeed, mode, 0);
        }

        long _initialSeed;
        StartMode _initialMode;

        void CreateAgents(long initialSeed)
        {
            _agents.Clear();
            for (int i = 0; i < numAgents; i++)
            {
                VehicleController ctrl = VehicleFactory.Create(vehicleConfig, simConfig, transform, "Car_" + i, carMaterial);
                var agent = ctrl.gameObject.AddComponent<RaceAgentCore>();
                agent.Initialize(i, TrackRuntime.Geometry, ctrl, simConfig, vehicleConfig, rewardConfig);
                _agents.Add(agent);
                _rngs[i] = new DeterministicRng(initialSeed + i);
            }
        }

        /// <summary>
        /// Destroys and re-creates every car exactly as Initialize did. A teleport keeps the WheelColliders' internal
        /// PhysX state (suspension/contact/tire), so only fresh cars make an explicit reset reproducible within one
        /// process (M3 RESET). Not for ML-Agents scenes (Agent components live on the cars). Allocates.
        /// </summary>
        public void RebuildAgents()
        {
            for (int i = 0; i < _agents.Count; i++)
            {
                GameObject go = _agents[i].gameObject;
                go.SetActive(false); // leaves the physics scene now; Destroy completes at the end of the frame
                Destroy(go);
            }
            CreateAgents(_initialSeed);
            ResetAll(_initialSeed, _initialMode, 0);
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
            _agents[i].MaxLaps = MaxLaps;
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

        string _envConfigHash;
        /// <summary>C0.11: Sim + Vehicle + Reward + Track + obs layout. Frozen after the M2 DoD (C0.14).</summary>
        public string EnvConfigHash => _envConfigHash ?? (_envConfigHash = ComputeEnvConfigHash());

        /// <summary>env_config_hash over Sim, Vehicle, Reward, Track and the obs layout (plus optional extra parts).</summary>
        public string ComputeEnvConfigHash(params IHashableConfig[] extra) =>
            ComputeEnvConfigHash(simConfig, vehicleConfig, rewardConfig, trackDefinition, extra);

        /// <summary>Same hash without a running environment (EditMode tests, tools). Frozen value: contracts.md C0.14.</summary>
        public static string ComputeEnvConfigHash(SimConfig sim, VehicleConfig vehicle, RewardConfig reward, TrackDefinition track,
                                                  params IHashableConfig[] extra)
        {
            var parts = new List<IHashableConfig> { sim, vehicle, reward, track, new ObsLayoutPart() };
            if (extra != null) parts.AddRange(extra);
            return ConfigHash.Compute(parts.ToArray());
        }

        sealed class ObsLayoutPart : IHashableConfig
        {
            public void AppendCanonical(SortedDictionary<string, string> kv) => kv["obs.layoutHash"] = ObservationSpec.LayoutHash;
        }
    }
}
