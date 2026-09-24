using System.Collections.Generic;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>Simulation constants shared by all milestones (C0.3).</summary>
    [CreateAssetMenu(menuName = "Racing/Sim Config", fileName = "SimConfig")]
    public sealed class SimConfig : ScriptableObject, IHashableConfig
    {
        [Header("Timing")]
        public float fixedDeltaTime = 0.02f;
        public int decisionPeriod = 5;
        public int maxEpisodeDecisions = 3000;

        [Header("Episode signal timers (sim seconds)")]
        public float noProgressTimeout = 8f;
        public float flipTimeout = 1f;
        public float wrongWayTimeout = 2f;

        [Header("Environment")]
        public int numAgentsPerEnv = 16;

        [Header("WheelCollider substeps")]
        public float substepSpeedThreshold = 5f;
        public int substepsBelowThreshold = 12;
        public int substepsAboveThreshold = 15;

        public static SimConfig CreateDefault() => CreateInstance<SimConfig>();

        public void AppendCanonical(SortedDictionary<string, string> kv)
        {
            kv["sim.fixedDeltaTime"] = ConfigHash.F(fixedDeltaTime);
            kv["sim.decisionPeriod"] = ConfigHash.I(decisionPeriod);
            kv["sim.maxEpisodeDecisions"] = ConfigHash.I(maxEpisodeDecisions);
            kv["sim.noProgressTimeout"] = ConfigHash.F(noProgressTimeout);
            kv["sim.flipTimeout"] = ConfigHash.F(flipTimeout);
            kv["sim.wrongWayTimeout"] = ConfigHash.F(wrongWayTimeout);
            kv["sim.substepSpeedThreshold"] = ConfigHash.F(substepSpeedThreshold);
            kv["sim.substepsBelowThreshold"] = ConfigHash.I(substepsBelowThreshold);
            kv["sim.substepsAboveThreshold"] = ConfigHash.I(substepsAboveThreshold);
        }
    }
}
