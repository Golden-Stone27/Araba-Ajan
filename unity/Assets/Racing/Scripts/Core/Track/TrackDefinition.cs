using System;
using System.Collections.Generic;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>Validator character profile (M6). Metadata only: not part of env_config_hash.</summary>
    public enum TrackProfile
    {
        Benchmark = 0,
        Technical = 1,
        Speedway = 2,
        Elevation = 3,
        Procedural = 4
    }

    /// <summary>Road surface collider (M6). GroundBox = the M1 ground cube at y = 0 (flat tracks only).</summary>
    public enum RoadColliderMode
    {
        GroundBox = 0,
        MeshStrip = 1
    }

    /// <summary>Elevation key: height y (m) at u = s / L (horizontal arc-length fraction, 0 ≤ u &lt; 1).</summary>
    [Serializable]
    public struct ElevationKey
    {
        public float u;
        public float y;

        public ElevationKey(float u, float y)
        {
            this.u = u;
            this.y = y;
        }
    }

    /// <summary>
    /// Track source data: closed centripetal Catmull-Rom through control points (x, z), optional elevation profile.
    /// Hash rule (C0.20): fields added after the M2 Env Freeze enter the canonical JSON only when they differ
    /// from their defaults, so Track_A keeps its frozen env_config_hash.
    /// </summary>
    [CreateAssetMenu(menuName = "Racing/Track Definition", fileName = "TrackDefinition")]
    public sealed class TrackDefinition : ScriptableObject, IHashableConfig
    {
        public string trackId = "Track_A";
        [Tooltip("Validator character profile; metadata, not hashed.")]
        public TrackProfile profile = TrackProfile.Benchmark;
        [TextArea] public string notes;
        public Vector2[] controlPoints = TrackPresets.TrackA;
        public float width = 12f;
        public float wallHeight = 1.5f;
        public float wallThickness = 1f;
        public float wallSegmentLength = 2f;
        public float checkpointSpacing = 10f;
        public float sampleSpacing = 1f;
        public float catmullAlpha = 0.5f;

        [Header("M6: elevation and colliders (hashed only when non-default)")]
        [Tooltip("Periodic cosine-interpolated height profile, keys sorted by u in [0, 1). Empty = flat (y = 0).")]
        public ElevationKey[] elevation = new ElevationKey[0];
        [Tooltip("Wall colliders extend this far below the road (invisible) so the horizontal rays hit them on grades.")]
        public float wallColliderExtraBelow;
        [Tooltip("Wall colliders extend this far above wallHeight (invisible).")]
        public float wallColliderExtraAbove;
        public RoadColliderMode roadCollider = RoadColliderMode.GroundBox;
        [Tooltip("MeshStrip: road collider half-width = width/2 + wallThickness + margin.")]
        public float roadColliderMargin = 3f;

        public bool HasElevation => elevation != null && elevation.Length > 0;

        public static TrackDefinition CreateDefault() => CreateInstance<TrackDefinition>();

        /// <summary>
        /// Height at u = s / L. Between consecutive keys (wrapping from the last key to the first + 1):
        /// y = y0 + (y1 − y0)·(1 − cos πt)/2, so the grade is 0 at every key and C1-continuous.
        /// Max grade on a span = π/2 · |Δy| / Δs.
        /// </summary>
        public double HeightAt(double u)
        {
            if (!HasElevation) return 0.0;
            int n = elevation.Length;
            if (n == 1) return elevation[0].y;
            u -= Math.Floor(u);
            int i = n - 1;
            for (int k = 0; k < n; k++)
                if (elevation[k].u > u) { i = k - 1; break; }
            double u0, u1;
            ElevationKey a, b;
            if (i < 0)
            {
                a = elevation[n - 1]; b = elevation[0];
                u0 = a.u - 1.0; u1 = b.u;
            }
            else
            {
                a = elevation[i]; b = elevation[(i + 1) % n];
                u0 = a.u; u1 = i + 1 < n ? b.u : b.u + 1.0;
            }
            double t = (u - u0) / (u1 - u0);
            return a.y + (b.y - a.y) * 0.5 * (1.0 - Math.Cos(Math.PI * t));
        }

        public void AppendCanonical(SortedDictionary<string, string> kv)
        {
            kv["track.id"] = trackId;
            kv["track.width"] = ConfigHash.F(width);
            kv["track.wallHeight"] = ConfigHash.F(wallHeight);
            kv["track.wallThickness"] = ConfigHash.F(wallThickness);
            kv["track.wallSegmentLength"] = ConfigHash.F(wallSegmentLength);
            kv["track.checkpointSpacing"] = ConfigHash.F(checkpointSpacing);
            kv["track.sampleSpacing"] = ConfigHash.F(sampleSpacing);
            kv["track.catmullAlpha"] = ConfigHash.F(catmullAlpha);
            var sb = new System.Text.StringBuilder();
            foreach (var p in controlPoints) sb.Append(ConfigHash.F(p.x)).Append(',').Append(ConfigHash.F(p.y)).Append(';');
            kv["track.controlPoints"] = ConfigHash.Sha256Hex16(sb.ToString());

            // M6 fields: only when non-default (C0.20), so pre-M6 tracks keep their hash
            if (HasElevation)
            {
                var e = new System.Text.StringBuilder();
                foreach (ElevationKey k in elevation) e.Append(ConfigHash.F(k.u)).Append(',').Append(ConfigHash.F(k.y)).Append(';');
                kv["track.elevation"] = ConfigHash.Sha256Hex16(e.ToString());
            }
            if (wallColliderExtraBelow != 0f) kv["track.wallColliderExtraBelow"] = ConfigHash.F(wallColliderExtraBelow);
            if (wallColliderExtraAbove != 0f) kv["track.wallColliderExtraAbove"] = ConfigHash.F(wallColliderExtraAbove);
            if (roadCollider != RoadColliderMode.GroundBox)
            {
                kv["track.roadCollider"] = roadCollider.ToString();
                kv["track.roadColliderMargin"] = ConfigHash.F(roadColliderMargin);
            }
        }
    }
}
