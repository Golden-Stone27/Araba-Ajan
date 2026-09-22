using UnityEngine;

namespace Racing.Core
{
    public interface IVehicleInputSource
    {
        VehicleAction ReadAction();
    }

    /// <summary>Legacy Input Manager axes: Horizontal = steer, Vertical = throttle/brake.</summary>
    public sealed class KeyboardInputSource : IVehicleInputSource
    {
        public VehicleAction ReadAction() => new VehicleAction(Input.GetAxis("Horizontal"), Input.GetAxis("Vertical"));
    }

    public sealed class ZeroInputSource : IVehicleInputSource
    {
        public VehicleAction ReadAction() => VehicleAction.Zero;
    }

    /// <summary>
    /// Reference driver for automated tests: pure pursuit on the centreline plus a curvature-based
    /// speed profile with a braking look-ahead. Reads the agent state from the last physics step.
    /// </summary>
    public sealed class PurePursuitInputSource : IVehicleInputSource
    {
        public float LateralAccel = 8f;
        public float BrakeDecel = 6f;
        public float LookAheadGain = 0.8f, LookAheadMin = 8f, LookAheadMax = 25f;
        public float SpeedGain = 0.5f;
        public float BrakeWindow = 80f;

        readonly RaceAgentCore _agent;

        public PurePursuitInputSource(RaceAgentCore agent) { _agent = agent; }

        public VehicleAction ReadAction()
        {
            VehicleState st = _agent.State;
            TrackProjection proj = _agent.Projection;
            ITrack track = _agent.Track;
            VehicleController ctrl = _agent.Controller;
            VehicleConfig cfg = ctrl.Config;

            float v = st.LocalVelocity.z;
            float ld = Mathf.Clamp(LookAheadGain * v, LookAheadMin, LookAheadMax);
            Vector3 target = track.PointAt(proj.S + ld);
            Vector3 toTarget = target - st.Position;
            toTarget.y = 0f;
            Vector3 fwd = st.Rotation * Vector3.forward;
            fwd.y = 0f;
            float alpha = Vector3.SignedAngle(fwd, toTarget, Vector3.up) * Mathf.Deg2Rad;
            float deltaDeg = Mathf.Atan2(2f * cfg.WheelBase * Mathf.Sin(alpha), ld) * Mathf.Rad2Deg;
            float steer = Mathf.Clamp(deltaDeg / ctrl.MaxSteerDeg(v), -1f, 1f);

            float vTarget = cfg.speedCap;
            for (float d = 0f; d <= BrakeWindow; d += 2f)
            {
                float vc = Mathf.Min(cfg.speedCap, Mathf.Sqrt(LateralAccel * track.RadiusAt(proj.S + d)));
                vTarget = Mathf.Min(vTarget, Mathf.Sqrt(vc * vc + 2f * BrakeDecel * d));
            }
            float throttle = Mathf.Clamp(SpeedGain * (vTarget - v), -1f, 1f);
            return new VehicleAction(steer, throttle);
        }
    }
}
