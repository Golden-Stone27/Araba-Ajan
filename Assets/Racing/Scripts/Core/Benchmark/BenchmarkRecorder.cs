using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>
    /// C0.10 eval metrics. Observes every agent after each PhysicsStep and records only its FIRST episode.
    /// Engine-agnostic: the ML-Agents BenchmarkRunner and the bridge evaluator (M5) feed it the same way.
    /// </summary>
    public sealed class BenchmarkRecorder
    {
        public const string Schema = "race-benchmark/v1";

        sealed class Episode
        {
            public bool Done;
            public TermReason Reason;
            public readonly List<float> Laps = new List<float>();
            public readonly List<float[]> Sectors = new List<float[]>();
            public float[] CurrentSectors = new float[3];
            public int SectorMark; // sector boundaries passed in the current lap (0..2)
            public int PhysicsSteps;
            public double SpeedSum;
            public double SteerDeltaSum;
            public float PrevSteer;
        }

        readonly RaceEnvironment _env;
        readonly Episode[] _eps;
        readonly int _maxLaps;

        public int Episodes => _eps.Length;
        public int Finished { get; private set; }
        public bool AllDone => Finished == _eps.Length;

        public BenchmarkRecorder(RaceEnvironment env, int maxLaps)
        {
            _env = env;
            _maxLaps = maxLaps;
            _eps = new Episode[env.Agents.Count];
            for (int i = 0; i < _eps.Length; i++) _eps[i] = new Episode();
        }

        /// <summary>Call right after env.PhysicsStep(), before any agent is reset.</summary>
        public void Observe(AgentStepResult[] results)
        {
            float L = _env.Track.Length;
            for (int i = 0; i < _eps.Length; i++)
            {
                Episode e = _eps[i];
                if (e.Done) continue;
                RaceAgentCore a = _env.Agents[i];
                e.PhysicsSteps++;
                e.SpeedSum += a.State.IsFinite ? a.State.Velocity.magnitude : 0f;
                float steer = a.Controller.LastApplied.Steer;
                e.SteerDeltaSum += Mathf.Abs(steer - e.PrevSteer);
                e.PrevSteer = steer;

                // Sector split points at L/3 and 2L/3 (timer runs from the spawn / last start-line crossing).
                if (a.LapTimer.Running && e.SectorMark < 2)
                {
                    float s = a.Projection.S;
                    float boundary = (e.SectorMark + 1) * L / 3f;
                    if (s >= boundary && s < boundary + 50f)
                    {
                        e.CurrentSectors[e.SectorMark] = a.LapTimer.Current;
                        e.SectorMark++;
                    }
                }

                AgentStepResult r = results[i];
                if (r.LapCompleted)
                {
                    e.Laps.Add(r.LapTime);
                    var sec = new float[3];
                    sec[0] = e.CurrentSectors[0];
                    sec[1] = e.CurrentSectors[1] - e.CurrentSectors[0];
                    sec[2] = r.LapTime - e.CurrentSectors[1];
                    e.Sectors.Add(e.SectorMark == 2 ? sec : null);
                    e.SectorMark = 0;
                    e.CurrentSectors = new float[3];
                }
                if (r.Terminated || r.Truncated)
                {
                    e.Done = true;
                    e.Reason = r.Reason;
                    Finished++;
                }
            }
        }

        public bool Succeeded(int i) => _eps[i].Done && _eps[i].Reason == TermReason.Finished && _eps[i].Laps.Count >= _maxLaps;

        /// <summary>One "per_seed" entry of the C0.10 schema (JSON object text, invariant culture).</summary>
        public string PerSeedJson(int trainSeed)
        {
            var flying = new List<float>();
            var lap1 = new List<float>();
            var s1 = new List<float>();
            var s2 = new List<float>();
            var s3 = new List<float>();
            var reasons = new SortedDictionary<string, int>(StringComparer.Ordinal);
            double speedSum = 0, steerSum = 0;
            int speedN = 0, decisions = 0, success = 0;
            int K = Mathf.Max(1, _env.Sim.decisionPeriod);

            for (int i = 0; i < _eps.Length; i++)
            {
                Episode e = _eps[i];
                string key = ReasonName(e.Done ? e.Reason : TermReason.None);
                reasons.TryGetValue(key, out int c);
                reasons[key] = c + 1;
                speedSum += e.SpeedSum;
                speedN += e.PhysicsSteps;
                steerSum += e.SteerDeltaSum;
                decisions += e.PhysicsSteps / K;
                if (!Succeeded(i)) continue;
                success++;
                lap1.Add(e.Laps[0]);
                for (int l = 1; l < e.Laps.Count; l++)
                {
                    flying.Add(e.Laps[l]);
                    float[] sec = e.Sectors[l];
                    if (sec == null) continue;
                    s1.Add(sec[0]);
                    s2.Add(sec[1]);
                    s3.Add(sec[2]);
                }
            }

            var sb = new StringBuilder();
            sb.Append('{');
            Kv(sb, "seed", trainSeed).Append(',');
            Kv(sb, "episodes", _eps.Length).Append(',');
            Kv(sb, "completion_rate", _eps.Length > 0 ? (float)success / _eps.Length : 0f).Append(',');
            Kv(sb, "flying_lap_median_s", Median(flying)).Append(',');
            Kv(sb, "flying_lap_best_s", flying.Count > 0 ? Min(flying) : float.NaN).Append(',');
            Kv(sb, "lap1_median_s", Median(lap1)).Append(',');
            Kv(sb, "mean_speed_mps", speedN > 0 ? (float)(speedSum / speedN) : 0f).Append(',');
            Kv(sb, "steer_smoothness", decisions > 0 ? (float)(steerSum / decisions) : 0f).Append(',');
            sb.Append("\"term_reasons\":{");
            bool first = true;
            foreach (var kv in reasons)
            {
                if (!first) sb.Append(',');
                first = false;
                Kv(sb, kv.Key, kv.Value);
            }
            sb.Append("},\"sector_times_s\":[").Append(Num(Median(s1))).Append(',').Append(Num(Median(s2))).Append(',').Append(Num(Median(s3))).Append(']');
            sb.Append(",\"flying_laps_s\":[");
            for (int i = 0; i < flying.Count; i++) sb.Append(i > 0 ? "," : "").Append(Num(flying[i]));
            sb.Append("]}");
            return sb.ToString();
        }

        /// <summary>Standalone single-policy report (one per trained seed), later merged into the baseline file.</summary>
        public string ReportJson(string policy, string evaluator, int trainSeed, string buildId)
        {
            var sb = new StringBuilder();
            sb.Append('{');
            Kv(sb, "schema", Schema).Append(',');
            Kv(sb, "policy", policy).Append(',');
            Kv(sb, "evaluator", evaluator).Append(',');
            Kv(sb, "env_config_hash", _env.EnvConfigHash).Append(',');
            Kv(sb, "obs_layout_hash", ObservationSpec.LayoutHash).Append(',');
            Kv(sb, "unity", Application.unityVersion).Append(',');
            Kv(sb, "build_id", buildId).Append(',');
            sb.Append("\"eval\":{");
            Kv(sb, "episodes", _eps.Length).Append(',');
            Kv(sb, "seed_base", _env.Seed).Append(',');
            Kv(sb, "laps", _maxLaps).Append(',');
            Kv(sb, "start", "grid").Append(',');
            sb.Append("\"deterministic\":true},");
            sb.Append("\"per_seed\":[").Append(PerSeedJson(trainSeed)).Append("]}");
            return sb.ToString();
        }

        public void Write(string path, string json)
        {
            string dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            File.WriteAllText(path, json + "\n", new UTF8Encoding(false));
        }

        public static string ReasonName(TermReason r)
        {
            switch (r)
            {
                case TermReason.Wall: return "wall";
                case TermReason.WrongWay: return "wrong_way";
                case TermReason.Stuck: return "stuck";
                case TermReason.Flip: return "flip";
                case TermReason.OutOfBounds: return "out_of_bounds";
                case TermReason.TimeLimit: return "time_limit";
                case TermReason.PhysicsError: return "physics_error";
                case TermReason.Finished: return "finished";
                default: return "none";
            }
        }

        static float Median(List<float> v)
        {
            if (v.Count == 0) return float.NaN;
            var c = new List<float>(v);
            c.Sort();
            int n = c.Count;
            return n % 2 == 1 ? c[n / 2] : 0.5f * (c[n / 2 - 1] + c[n / 2]);
        }

        static float Min(List<float> v)
        {
            float m = float.PositiveInfinity;
            foreach (float x in v) m = Mathf.Min(m, x);
            return m;
        }

        /// <summary>JSON number; NaN/Inf become null.</summary>
        public static string Num(float v) => float.IsFinite(v) ? v.ToString("R", CultureInfo.InvariantCulture) : "null";

        static StringBuilder Kv(StringBuilder sb, string k, float v) => sb.Append('"').Append(k).Append("\":").Append(Num(v));
        static StringBuilder Kv(StringBuilder sb, string k, long v) => sb.Append('"').Append(k).Append("\":").Append(v.ToString(CultureInfo.InvariantCulture));
        static StringBuilder Kv(StringBuilder sb, string k, string v) => sb.Append('"').Append(k).Append("\":\"").Append(v).Append('"');
    }
}
