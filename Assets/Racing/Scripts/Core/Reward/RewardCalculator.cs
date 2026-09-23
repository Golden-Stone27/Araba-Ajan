using UnityEngine;

namespace Racing.Core
{
    /// <summary>Everything the reward needs from one physics step. Tests build it synthetically.</summary>
    public struct StepContext
    {
        public Vector3 Velocity;
        /// <summary>Centreline tangent t̂(s) at the car's projection (checkpoint order direction).</summary>
        public Vector3 Tangent;
        public float WallClearance;
        /// <summary>Applied (sanitised) steer of this step and of the previous step.</summary>
        public float Steer, PrevSteer;
        public CheckpointEvent Cp;
        public TermReason Reason;
    }

    /// <summary>Per-component reward of one step (or an accumulated episode). Each part is logged separately.</summary>
    public struct RewardBreakdown
    {
        public float Speed, Wall, Smooth, Checkpoint, Lap, Terminal;

        public float Total => Speed + Wall + Smooth + Checkpoint + Lap + Terminal;

        public void Add(in RewardBreakdown o)
        {
            Speed += o.Speed;
            Wall += o.Wall;
            Smooth += o.Smooth;
            Checkpoint += o.Checkpoint;
            Lap += o.Lap;
            Terminal += o.Terminal;
        }
    }

    /// <summary>
    /// M2 reward, per physics step j (K = decision period, decision reward R_t = Σ r_j):
    /// r_j = (w_v/K)·clip(v·t̂/50, −1, 1) − (w_wall/K)·max(0, 1 − clearance/1.5) − w_smooth·|a0_j − a0_{j−1}|
    ///       + w_cp·1[Passed] + w_lap·1[LapCompleted] + R_term(reason).
    /// PhysicsError steps give exactly 0 (the state may be NaN).
    /// </summary>
    public sealed class RewardCalculator
    {
        readonly RewardConfig _cfg;
        readonly float _invK;

        public RewardConfig Config => _cfg;

        public RewardCalculator(RewardConfig cfg, int decisionPeriod)
        {
            _cfg = cfg;
            _invK = 1f / Mathf.Max(1, decisionPeriod);
        }

        public RewardBreakdown Step(in StepContext c)
        {
            var r = new RewardBreakdown();
            if (c.Reason == TermReason.PhysicsError) return r;

            float vt = Vector3.Dot(c.Velocity, c.Tangent);
            r.Speed = _cfg.wSpeed * _invK * Mathf.Clamp(vt / _cfg.speedNorm, -1f, 1f);
            r.Wall = -_cfg.wWall * _invK * Mathf.Max(0f, 1f - c.WallClearance / _cfg.wallClearanceRef);
            r.Smooth = -_cfg.wSmooth * Mathf.Abs(c.Steer - c.PrevSteer);
            if (c.Cp == CheckpointEvent.Passed) r.Checkpoint = _cfg.wCheckpoint;
            else if (c.Cp == CheckpointEvent.LapCompleted) r.Lap = _cfg.wLap;
            r.Terminal = _cfg.TerminalReward(c.Reason);
            return r;
        }
    }
}
