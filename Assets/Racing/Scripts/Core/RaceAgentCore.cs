using UnityEngine;

namespace Racing.Core
{
    /// <summary>One car = one agent. Engine-agnostic: ML-Agents (M2) and the bridge (M3) drive it the same way (C0.8).</summary>
    public sealed class RaceAgentCore : MonoBehaviour
    {
        SimConfig _sim;
        VehicleConfig _vehicleConfig;
        TrackGeometry _track;
        VehicleController _controller;
        CollisionSensor _collisions;
        readonly RaySensor _rays = new RaySensor();
        CheckpointTracker _tracker;
        LapTimer _lapTimer;
        EpisodeMonitor _monitor;
        RewardCalculator _reward;
        TerminationPolicy _termination;

        VehicleState _state;
        TrackProjection _projection;
        Vector3 _prevPos;
        int _physicsSteps;
        double _episodeReturn; // double: float32 accumulation over 15k steps drifts ~1e-3 from the per-block sum (M3 DoD 5)
        float _prevSteer;
        RewardBreakdown _episodeRewards;
        AgentTelemetry _telemetry;

        public int Index { get; private set; }
        public ITrack Track => _track;
        public TrackGeometry Geometry => _track;
        public VehicleController Controller => _controller;
        public CollisionSensor Collisions => _collisions;
        public RaySensor Rays => _rays;
        public CheckpointTracker Tracker => _tracker;
        public LapTimer LapTimer => _lapTimer;
        public VehicleState State => _state;
        public TrackProjection Projection => _projection;
        public EpisodeSignals LastSignals { get; private set; }
        public AgentTelemetry Telemetry => _telemetry;
        public int PhysicsSteps => _physicsSteps;
        public IVehicleInputSource InputSource { get; set; }
        /// <summary>Eval lap target (Finished truncation); 0 = unlimited. Set by RaceEnvironment.</summary>
        public int MaxLaps { get; set; }
        public RewardConfig RewardConfig => _reward.Config;
        /// <summary>Per-component reward accumulated since BeginEpisode.</summary>
        public RewardBreakdown EpisodeRewards => _episodeRewards;
        public RewardBreakdown LastRewards { get; private set; }

        public void Initialize(int index, TrackGeometry track, VehicleController controller, SimConfig sim, VehicleConfig vehicleConfig,
                               RewardConfig rewardConfig)
        {
            Index = index;
            _track = track;
            _controller = controller;
            _collisions = controller.GetComponent<CollisionSensor>();
            _sim = sim;
            _vehicleConfig = vehicleConfig;
            _tracker = new CheckpointTracker(track);
            _lapTimer = new LapTimer(sim.fixedDeltaTime);
            _monitor = new EpisodeMonitor(sim, vehicleConfig.halfWidthForClearance);
            _reward = new RewardCalculator(rewardConfig != null ? rewardConfig : RewardConfig.CreateDefault(), sim.decisionPeriod);
            _termination = new TerminationPolicy(sim);
        }

        public void BeginEpisode(in SpawnSpec spawn)
        {
            Pose pose = _track.SpawnPose(spawn.S, spawn.Lateral, spawn.HeadingOffsetDeg);
            pose.position += Vector3.up * _vehicleConfig.spawnHeight;
            _controller.TeleportTo(pose, spawn.InitialSpeed);
            _collisions.ClearStep();

            _tracker.Reset(_track.NextGateAfter(spawn.S), spawn.StartTiming);
            _lapTimer.Reset();
            if (spawn.StartTiming) _lapTimer.Start();
            _monitor.Reset();
            _physicsSteps = 0;
            _episodeReturn = 0.0;
            _prevSteer = 0f;
            _episodeRewards = default;
            LastRewards = default;

            _state = _controller.ReadState();
            _projection = _track.Project(_state.Position, -1);
            _prevPos = _state.Position;
            LastSignals = default;
            _telemetry = default;
            UpdateTelemetry(false);
        }

        /// <summary>Pre-physics (C0.8).</summary>
        public void ApplyAction(in VehicleAction action) => _controller.ApplyAction(action, _sim.fixedDeltaTime);

        /// <summary>Called by RaceEnvironment right before Physics.Simulate.</summary>
        public void BeforePhysicsStep() => _collisions.ClearStep();

        /// <summary>
        /// Post-physics (C0.8): progress, checkpoints, lap timing, signals → TerminationPolicy + RewardCalculator (M2).
        /// The core never resets itself; the driver (ML-Agents / bridge / sandbox) decides what to do with Terminated/Truncated.
        /// </summary>
        public AgentStepResult AfterPhysicsStep()
        {
            _physicsSteps++;
            _state = _controller.ReadState();
            _projection = _track.Project(_state.Position, _state.IsFinite ? _projection.SampleIndex : -1);

            _lapTimer.Tick();
            CheckpointEvent cp = _tracker.Update(_prevPos, _state.Position);
            bool lapDone = false;
            float lapTime = float.NaN;
            if (cp == CheckpointEvent.LapCompleted)
            {
                lapTime = _lapTimer.MarkLap();
                lapDone = true;
            }
            else if (cp == CheckpointEvent.Passed && _tracker.LastPassedIndex == 0)
            {
                _lapTimer.Start();
            }
            _prevPos = _state.Position;

            EpisodeSignals sig = _monitor.Evaluate(_state, _projection, _track, _collisions.WallContactThisStep, cp);
            LastSignals = sig;

            TermReason reason = _termination.Evaluate(sig, _physicsSteps, _tracker.LapsCompleted, MaxLaps);
            float steer = _controller.LastApplied.Steer;
            var ctx = new StepContext
            {
                Velocity = _state.Velocity,
                Tangent = _projection.Tangent,
                WallClearance = sig.WallClearance,
                Steer = steer,
                PrevSteer = _prevSteer,
                Cp = cp,
                Reason = reason
            };
            _prevSteer = steer;
            RewardBreakdown rb = _reward.Step(ctx);
            LastRewards = rb;
            _episodeRewards.Add(rb);
            float r = rb.Total;
            AddReward(r);
            if (reason != TermReason.None) SetTermination(reason);

            var result = new AgentStepResult
            {
                Reward = r,
                Terminated = TerminationPolicy.IsTermination(reason),
                Truncated = TerminationPolicy.IsTruncation(reason),
                Reason = reason,
                Signals = sig,
                LapCompleted = lapDone,
                LapTime = lapTime
            };
            UpdateTelemetry(lapDone);
            return result;
        }

        /// <summary>M2+: accumulates the C#-side episode return reported in RACE_INFO.</summary>
        public void AddReward(float r)
        {
            _episodeReturn += r;
            _telemetry.EpisodeReturn = (float)_episodeReturn;
        }

        public void SetTermination(TermReason reason) => _telemetry.TermReason = reason;

        public void WriteObservation(float[] dst, int offset) =>
            ObservationBuilder.Write(dst, offset, _state, _projection, _track, _rays, _controller.LastApplied);

        void UpdateTelemetry(bool lapDone)
        {
            _telemetry.LapCompletedThisStep = lapDone;
            _telemetry.Laps = _tracker.LapsCompleted;
            _telemetry.NextCheckpoint = _tracker.NextIndex;
            _telemetry.EpisodeDecisions = _physicsSteps / Mathf.Max(1, _sim.decisionPeriod);
            _telemetry.LastLapTime = _lapTimer.Last;
            _telemetry.BestLapTime = _lapTimer.Best;
            _telemetry.Speed = _state.Velocity.magnitude;
            _telemetry.Progress = _projection.S / _track.Length;
            _telemetry.EpisodeReturn = (float)_episodeReturn;
            _telemetry.PosX = _state.Position.x;
            _telemetry.PosZ = _state.Position.z;
        }

        void OnDrawGizmosSelected()
        {
            if (_track == null) return;
            for (int i = 0; i < RaySensor.Count; i++)
            {
                float d = _rays.LastNormalized(i);
                Vector3 dir = Quaternion.Euler(0f, _rays.LastYawDeg + RaySensor.AngleDeg(i), 0f) * Vector3.forward;
                Gizmos.color = d < 1f ? Color.red : Color.green;
                Gizmos.DrawLine(_rays.LastOrigin, _rays.LastOrigin + dir * d * RaySensor.MaxDistance);
            }
        }
    }
}
