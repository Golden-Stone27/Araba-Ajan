using System.Collections.Generic;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>Reward weights (M2). Part of env_config_hash; frozen after the M2 DoD (C0.11, C0.14).</summary>
    [CreateAssetMenu(menuName = "Racing/Reward Config", fileName = "RewardConfig")]
    public sealed class RewardConfig : ScriptableObject, IHashableConfig
    {
        [Header("Dense terms (per decision, spread over K physics steps)")]
        public float wSpeed = 0.1f;
        public float speedNorm = 50f;
        public float wWall = 0.05f;
        public float wallClearanceRef = 1.5f;

        [Header("Steering smoothness (charged once per decision boundary)")]
        public float wSmooth = 0.02f;

        [Header("Sparse terms")]
        public float wCheckpoint = 0.05f;
        public float wLap = 2f;

        [Header("Terminal (C0.9)")]
        public float crashPenalty = -1f;
        public float stuckPenalty = -0.5f;

        public static RewardConfig CreateDefault() => CreateInstance<RewardConfig>();

        /// <summary>R_term(reason). Truncations (TimeLimit, Finished) and PhysicsError give 0.</summary>
        public float TerminalReward(TermReason reason)
        {
            switch (reason)
            {
                case TermReason.Wall:
                case TermReason.WrongWay:
                case TermReason.Flip:
                case TermReason.OutOfBounds:
                    return crashPenalty;
                case TermReason.Stuck:
                    return stuckPenalty;
                default:
                    return 0f;
            }
        }

        public void AppendCanonical(SortedDictionary<string, string> kv)
        {
            kv["reward.wSpeed"] = ConfigHash.F(wSpeed);
            kv["reward.speedNorm"] = ConfigHash.F(speedNorm);
            kv["reward.wWall"] = ConfigHash.F(wWall);
            kv["reward.wallClearanceRef"] = ConfigHash.F(wallClearanceRef);
            kv["reward.wSmooth"] = ConfigHash.F(wSmooth);
            kv["reward.wCheckpoint"] = ConfigHash.F(wCheckpoint);
            kv["reward.wLap"] = ConfigHash.F(wLap);
            kv["reward.crashPenalty"] = ConfigHash.F(crashPenalty);
            kv["reward.stuckPenalty"] = ConfigHash.F(stuckPenalty);
        }
    }
}
