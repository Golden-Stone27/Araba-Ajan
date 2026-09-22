using UnityEngine;

namespace Racing.Core
{
    /// <summary>
    /// M1 "Manual" driver for the Sandbox scene: every FixedUpdate applies keyboard (agent 0) or
    /// pure-pursuit input, then env.PhysicsStep(). Keys: P autopilot, R reset, H HUD.
    /// Resets only on flip / out-of-bounds / non-finite (and stuck for autopilot cars); wall hits
    /// are shown on the HUD. Real termination rules belong to M2.
    /// </summary>
    [DefaultExecutionOrder(-800)]
    public sealed class SimulationDriver : MonoBehaviour
    {
        [SerializeField] RaceEnvironment environment;
        [SerializeField] bool autopilot;

        KeyboardInputSource _keyboard;
        PurePursuitInputSource[] _autopilots;

        public RaceEnvironment Environment => environment;
        public bool Autopilot => autopilot;

        void Start()
        {
            if (environment == null) environment = FindAnyObjectByType<RaceEnvironment>();
            _keyboard = new KeyboardInputSource();
            _autopilots = new PurePursuitInputSource[environment.Agents.Count];
            for (int i = 0; i < _autopilots.Length; i++) _autopilots[i] = new PurePursuitInputSource(environment.Agents[i]);
        }

        void Update()
        {
            if (Input.GetKeyDown(KeyCode.P)) autopilot = !autopilot;
            if (Input.GetKeyDown(KeyCode.R)) environment.ResetAgent(0);
        }

        void FixedUpdate()
        {
            if (environment == null || !environment.IsInitialized) return;
            var agents = environment.Agents;
            for (int i = 0; i < agents.Count; i++)
            {
                VehicleAction a = (i == 0 && !autopilot) ? _keyboard.ReadAction() : _autopilots[i].ReadAction();
                agents[i].ApplyAction(a);
            }
            AgentStepResult[] results = environment.PhysicsStep();
            for (int i = 0; i < results.Length; i++)
            {
                EpisodeSignals s = results[i].Signals;
                bool machine = i != 0 || autopilot;
                if (s.Flipped || s.OutOfBounds || s.NonFinite || (machine && s.NoProgressTimeout)) environment.ResetAgent(i);
            }
        }
    }
}
