using UnityEngine;

namespace Racing.Core
{
    /// <summary>Observation contract (C0.6). Changing anything here changes obs_layout_hash.</summary>
    public static class ObservationSpec
    {
        public const int Size = 26;
        public const string Layout =
            "RACE_OBS_V1|n=26|rays=15,fov=180,max=50,h=0.5|vfwd/50|vlat/50|yaw/3|psi1(sin,cos)@0|psi2(sin,cos)@30|elat/6|prev(steer,thr)|grounded/4";

        public const int RayStart = 0, VFwd = 15, VLat = 16, YawRate = 17, Psi1Sin = 18, Psi1Cos = 19,
                         Psi2Sin = 20, Psi2Cos = 21, ELat = 22, PrevSteer = 23, PrevThrottle = 24, Grounded = 25;

        public const float VMax = 50f, YawRateMax = 3f, LookAhead = 30f, LateralNorm = 6f, LateralClip = 1.5f;

        static readonly float[] s_low = BuildBounds(false), s_high = BuildBounds(true);
        public static float Low(int i) => s_low[i];
        public static float High(int i) => s_high[i];

        static string s_hash;
        public static string LayoutHash => s_hash ?? (s_hash = ConfigHash.Sha256Hex16(Layout));

        static float[] BuildBounds(bool high)
        {
            var b = new float[Size];
            for (int i = 0; i < Size; i++) b[i] = high ? 1f : -1f;
            for (int i = 0; i < RaySensor.Count; i++) b[RayStart + i] = high ? 1f : 0f;
            b[ELat] = high ? LateralClip : -LateralClip;
            b[Grounded] = high ? 1f : 0f;
            return b;
        }
    }

    /// <summary>15 horizontal (yaw-only) rays over 180°, Wall layer only, triggers ignored.</summary>
    public sealed class RaySensor
    {
        public const int Count = 15;
        public const float MaxDistance = 50f, Height = 0.5f, ForwardOffset = 2.1f, FovDeg = 180f;

        readonly float[] _last = new float[Count];
        public float LastNormalized(int i) => _last[i];
        public Vector3 LastOrigin { get; private set; }
        public float LastYawDeg { get; private set; }

        public static float AngleDeg(int i) => -0.5f * FovDeg + i * (FovDeg / (Count - 1));

        public static float YawDeg(Quaternion rotation)
        {
            Vector3 f = rotation * Vector3.forward;
            return Mathf.Atan2(f.x, f.z) * Mathf.Rad2Deg;
        }

        public void Sample(in VehicleState s, float[] dst, int offset)
        {
            float yaw = YawDeg(s.Rotation);
            Vector3 fwdFlat = Quaternion.Euler(0f, yaw, 0f) * Vector3.forward;
            Vector3 origin = s.Position + Vector3.up * Height + fwdFlat * ForwardOffset;
            LastOrigin = origin;
            LastYawDeg = yaw;
            for (int i = 0; i < Count; i++)
            {
                Vector3 dir = Quaternion.Euler(0f, yaw + AngleDeg(i), 0f) * Vector3.forward;
                float d = Physics.Raycast(origin, dir, out RaycastHit hit, MaxDistance, RacingLayers.WallMask, QueryTriggerInteraction.Ignore)
                    ? hit.distance / MaxDistance
                    : 1f;
                _last[i] = d;
                dst[offset + i] = d;
            }
        }
    }

    public static class ObservationBuilder
    {
        /// <summary>Writes the 26-float observation (C0.6). Allocation-free.</summary>
        public static void Write(float[] dst, int o, in VehicleState s, in TrackProjection proj, ITrack track,
                                 RaySensor rays, in VehicleAction previous)
        {
            rays.Sample(s, dst, o + ObservationSpec.RayStart);
            dst[o + ObservationSpec.VFwd] = Mathf.Clamp(s.LocalVelocity.z / ObservationSpec.VMax, -1f, 1f);
            dst[o + ObservationSpec.VLat] = Mathf.Clamp(s.LocalVelocity.x / ObservationSpec.VMax, -1f, 1f);
            dst[o + ObservationSpec.YawRate] = Mathf.Clamp(s.LocalAngularVelocity.y / ObservationSpec.YawRateMax, -1f, 1f);

            float psi1 = SignedHeadingErrorRad(s.Rotation, proj.Tangent);
            dst[o + ObservationSpec.Psi1Sin] = Mathf.Sin(psi1);
            dst[o + ObservationSpec.Psi1Cos] = Mathf.Cos(psi1);
            float psi2 = SignedHeadingErrorRad(s.Rotation, track.TangentAt(proj.S + ObservationSpec.LookAhead));
            dst[o + ObservationSpec.Psi2Sin] = Mathf.Sin(psi2);
            dst[o + ObservationSpec.Psi2Cos] = Mathf.Cos(psi2);

            dst[o + ObservationSpec.ELat] = Mathf.Clamp(proj.Lateral / ObservationSpec.LateralNorm, -ObservationSpec.LateralClip, ObservationSpec.LateralClip);
            dst[o + ObservationSpec.PrevSteer] = previous.Steer;
            dst[o + ObservationSpec.PrevThrottle] = previous.Throttle;
            dst[o + ObservationSpec.Grounded] = s.GroundedWheels * 0.25f;
        }

        /// <summary>Signed XZ angle from car forward to the tangent; positive = tangent to the right.</summary>
        public static float SignedHeadingErrorRad(Quaternion rotation, Vector3 tangent)
        {
            Vector3 f = rotation * Vector3.forward;
            f.y = 0f;
            tangent.y = 0f;
            if (f.sqrMagnitude < 1e-8f || tangent.sqrMagnitude < 1e-8f) return 0f;
            return Vector3.SignedAngle(f, tangent, Vector3.up) * Mathf.Deg2Rad;
        }
    }
}
