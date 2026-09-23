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

    /// <summary>
    /// Track source data: closed centripetal Catmull-Rom through control points (x, z).
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

        public static TrackDefinition CreateDefault() => CreateInstance<TrackDefinition>();

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
        }
    }
}
