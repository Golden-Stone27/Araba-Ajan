using Racing.Core;
using Unity.InferenceEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;
using ObservationSpec = Racing.Core.ObservationSpec;

namespace Racing.MLAgents
{
    /// <summary>
    /// ML-Agents adapter around one RaceAgentCore. Owns no game logic: observation, action, reward and
    /// termination all come from Core; MlaSimulationDriver runs the C0.8 step order.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class RaceAgent : Agent
    {
        public const string Behavior = "RaceCar";

        RaceEnvironment _env;
        RaceAgentCore _core;
        readonly float[] _obs = new float[ObservationSpec.Size];
        readonly KeyboardInputSource _keyboard = new KeyboardInputSource();

        public RaceAgentCore Core => _core;

        /// <summary>
        /// Adds BehaviorParameters → RaceAgent → DecisionRequester in that order: Agent.OnEnable reads the
        /// BehaviorParameters immediately and DecisionRequester.Awake needs the Agent.
        /// </summary>
        public static RaceAgent Attach(RaceAgentCore core, int decisionPeriod, ModelAsset model, BehaviorType type, bool deterministic)
        {
            GameObject go = core.gameObject;
            var bp = go.AddComponent<BehaviorParameters>();
            bp.BehaviorName = Behavior;
            bp.BrainParameters.VectorObservationSize = ObservationSpec.Size;
            bp.BrainParameters.NumStackedVectorObservations = 1;
            bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);
            bp.Model = model;
            bp.InferenceDevice = InferenceDevice.Burst;
            bp.BehaviorType = type;
            bp.DeterministicInference = deterministic;
            bp.UseChildSensors = false;
            bp.UseChildActuators = false;

            var agent = go.AddComponent<RaceAgent>();

            var dr = go.AddComponent<DecisionRequester>();
            dr.DecisionPeriod = decisionPeriod;
            dr.DecisionStep = 0;
            dr.TakeActionsBetweenDecisions = true;
            return agent;
        }

        public override void Initialize()
        {
            _core = GetComponent<RaceAgentCore>();
            _env = GetComponentInParent<RaceEnvironment>();
            MaxStep = 0; // truncation (TimeLimit) is decided in Core
        }

        public override void OnEpisodeBegin()
        {
            // RaceEnvironment.Initialize/ResetAll already spawned a fresh episode; the Academy's first forced
            // reset must not resample it (keeps eval spawns identical to the bridge evaluator).
            if (_core.PhysicsSteps > 0) _env.ResetAgent(_core.Index);
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            _core.WriteObservation(_obs, 0);
            sensor.AddObservation(_obs);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            var a = actions.ContinuousActions;
            _core.ApplyAction(new VehicleAction(a[0], a[1]));
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            VehicleAction a = _keyboard.ReadAction();
            var c = actionsOut.ContinuousActions;
            c[0] = a.Steer;
            c[1] = a.Throttle;
        }
    }
}
