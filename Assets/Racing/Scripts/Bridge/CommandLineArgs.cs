using System;
using System.Globalization;

namespace Racing.Bridge
{
    /// <summary>Player arguments: -bridgePort p, -numAgents N, -bridgeTimingLog path (all optional).</summary>
    public static class CommandLineArgs
    {
        public static string GetString(string name, string fallback = null)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (string.Equals(args[i], name, StringComparison.Ordinal)) return args[i + 1];
            return fallback;
        }

        public static int GetInt(string name, int fallback)
        {
            string v = GetString(name);
            return v != null && int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out int x) ? x : fallback;
        }
    }
}
