using System;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>
    /// Ordered list of the selectable tracks (M6, contracts C0.20). Index 0 is always the Benchmark track (Track_A);
    /// indices are stable once published (-trackIndex, HELLO track_index). Ids are ASCII and unique (Ordinal).
    /// </summary>
    [CreateAssetMenu(menuName = "Racing/Track Catalog", fileName = "TrackCatalog")]
    public sealed class TrackCatalog : ScriptableObject
    {
        public const string BenchmarkId = "Track_A";

        public TrackDefinition[] tracks = new TrackDefinition[0];

        public int Count => tracks.Length;

        public TrackDefinition Get(int index) => index >= 0 && index < tracks.Length ? tracks[index] : null;

        public int IndexOf(string trackId)
        {
            for (int i = 0; i < tracks.Length; i++)
                if (tracks[i] != null && string.Equals(tracks[i].trackId, trackId, StringComparison.Ordinal)) return i;
            return -1;
        }

        /// <summary>
        /// Resolves a catalog track by id or asset name (e.g. "TrackDefinition_B"), or generates "proc:&lt;seed&gt;"
        /// (ProceduralTrackGenerator; index = -1, not in the catalog).
        /// </summary>
        public bool TryResolve(string name, out TrackDefinition def, out int index)
        {
            if (ProceduralTrackGenerator.TryParseName(name, out long seed))
            {
                index = -1;
                def = ProceduralTrackGenerator.Generate(seed).Definition;
                return true;
            }
            index = IndexOf(name);
            if (index < 0)
                for (int i = 0; i < tracks.Length; i++)
                    if (tracks[i] != null && string.Equals(tracks[i].name, name, StringComparison.Ordinal)) { index = i; break; }
            def = index >= 0 ? tracks[index] : null;
            return def != null;
        }
    }
}
