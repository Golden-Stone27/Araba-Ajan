using System;
using System.Globalization;

namespace Racing.Bridge
{
    /// <summary>
    /// Player arguments: -bridgePort p, -numAgents N, -bridgeTimingLog path, -trackName id|asset|proc:seed, -trackIndex i
    /// (all optional). The string[] overloads take an explicit argument list (tests).
    /// </summary>
    public static class CommandLineArgs
    {
        public const string TrackName = "-trackName";
        public const string TrackIndex = "-trackIndex";

        public static string GetString(string name, string fallback = null) => GetString(Environment.GetCommandLineArgs(), name, fallback);

        public static int GetInt(string name, int fallback) => GetInt(Environment.GetCommandLineArgs(), name, fallback);

        public static string GetString(string[] args, string name, string fallback = null)
        {
            for (int i = 0; i < args.Length - 1; i++)
                if (string.Equals(args[i], name, StringComparison.Ordinal)) return args[i + 1];
            return fallback;
        }

        public static int GetInt(string[] args, string name, int fallback)
        {
            string v = GetString(args, name);
            return v != null && int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out int x) ? x : fallback;
        }

        public static bool Has(string[] args, string name)
        {
            for (int i = 0; i < args.Length; i++)
                if (string.Equals(args[i], name, StringComparison.Ordinal)) return true;
            return false;
        }

        /// <summary>
        /// -trackName / -trackIndex (M6). Absent flags give name = null and index = -1. A flag without a value or an
        /// index that is not a non-negative decimal integer is an error; resolving the name happens in BridgeDriver.
        /// </summary>
        public static bool TryParseTrackArgs(string[] args, out string trackName, out int trackIndex, out string error)
        {
            trackName = null;
            trackIndex = -1;
            error = null;
            if (Has(args, TrackName))
            {
                trackName = GetString(args, TrackName);
                if (string.IsNullOrEmpty(trackName))
                {
                    error = TrackName + " needs a value (track id, asset name or proc:<seed>)";
                    return false;
                }
            }
            if (Has(args, TrackIndex))
            {
                string v = GetString(args, TrackIndex);
                if (v == null || !int.TryParse(v, NumberStyles.None, CultureInfo.InvariantCulture, out trackIndex))
                {
                    trackIndex = -1;
                    error = TrackIndex + " needs a non-negative integer, got " + (v == null ? "nothing" : "'" + v + "'");
                    return false;
                }
            }
            return true;
        }
    }
}
