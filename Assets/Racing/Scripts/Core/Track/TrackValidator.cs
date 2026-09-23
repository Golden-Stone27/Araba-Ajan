using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>
    /// M1 track acceptance criteria (see M1 spec, "Validator"; Report.Passed = Benchmark profile) plus the M6
    /// universal safety rules and character profiles (CheckProfile, contracts C0.20).
    /// </summary>
    public static class TrackValidator
    {
        public const float MinLength = 900f, MaxLength = 1400f;
        public const float MinRadius = 12f;
        public const float CornerRadius = 150f;
        public const float StraightThresholdRadius = 250f;
        public const float MinStraight = 200f;
        public const float SeparationArc = 60f;
        public const float MaxAbsCoordinate = 500f;

        public struct Corner
        {
            public float StartS, EndS, MinRadius, TurnDeg;
            public int Sign;
        }

        public sealed class Report
        {
            public float Length, MinRadius, LongestStraight, MinSeparation, MaxAbsCoordinate, RequiredSeparation;
            public bool HasHairpin, HasFastCorner, HasChicane, HasLeft, HasRight;
            public int HairpinCount, ChicaneCount;
            /// <summary>Signed integral of curvature over the lap (deg): +-360 for a simple closed track.</summary>
            public float TotalTurnDeg;
            public List<Corner> Corners = new List<Corner>();

            public bool Passed =>
                Length >= MinLength && Length <= MaxLength && MinRadius >= TrackValidator.MinRadius &&
                MinSeparation > RequiredSeparation && LongestStraight >= MinStraight && HasHairpin &&
                HasFastCorner && HasChicane && HasLeft && HasRight && MaxAbsCoordinate < TrackValidator.MaxAbsCoordinate;

            public override string ToString()
            {
                var c = CultureInfo.InvariantCulture;
                return string.Format(c,
                    "L={0:F1} Rmin={1:F1} straight={2:F1} minSep={3:F1}(>{4:F1}) maxAbs={5:F1} hairpin={6} fast={7} chicane={8} left={9} right={10} corners={11} => {12}",
                    Length, MinRadius, LongestStraight, MinSeparation, RequiredSeparation, MaxAbsCoordinate,
                    HasHairpin, HasFastCorner, HasChicane, HasLeft, HasRight, Corners.Count, Passed ? "PASS" : "FAIL") +
                    string.Format(c, " | hairpins={0} chicanes={1} turn={2:F1}deg", HairpinCount, ChicaneCount, TotalTurnDeg);
            }
        }

        // M6 universal rules (every catalog and procedural track, C0.20)
        public const float UniversalMinLength = 600f, UniversalMaxLength = 1600f; // 3 laps must fit 300 s (C0.3)
        public const float MinWidth = 10f, MaxWidth = 14f;
        public const float TurnClosureToleranceDeg = 5f;
        public const float ProceduralMaxMinRadius = 40f;

        /// <summary>
        /// Universal safety rules and the definition's character profile. Returns the failed rules (empty = pass).
        /// Benchmark = the M1 criteria (Report.Passed), unchanged.
        /// </summary>
        public static List<string> CheckProfile(TrackDefinition def, Report r)
        {
            var fail = new List<string>();
            var c = CultureInfo.InvariantCulture;
            void Need(bool ok, string what, float value)
            {
                if (!ok) fail.Add(string.Format(c, "{0} ({1:F2})", what, value));
            }

            Need(r.Length >= UniversalMinLength && r.Length <= UniversalMaxLength, "length in [600, 1600]", r.Length);
            Need(r.MinRadius >= MinRadius, "min radius >= 12", r.MinRadius);
            Need(r.MinSeparation > r.RequiredSeparation, "self separation > W + 4", r.MinSeparation);
            Need(r.MaxAbsCoordinate < MaxAbsCoordinate, "|coordinate| < 500", r.MaxAbsCoordinate);
            Need(def.width >= MinWidth && def.width <= MaxWidth, "width in [10, 14]", def.width);
            Need(Mathf.Abs(Mathf.Abs(r.TotalTurnDeg) - 360f) <= TurnClosureToleranceDeg, "total turn +-360", r.TotalTurnDeg);

            switch (def.profile)
            {
                case TrackProfile.Benchmark:
                    Need(r.Passed, "benchmark (M1) criteria", 0f);
                    break;
                case TrackProfile.Technical:
                    Need(r.Corners.Count >= 8, "corners >= 8", r.Corners.Count);
                    Need(r.LongestStraight <= 150f, "longest straight <= 150", r.LongestStraight);
                    Need(r.HairpinCount >= 1, "hairpins >= 1", r.HairpinCount);
                    Need(r.ChicaneCount >= 2, "chicanes >= 2", r.ChicaneCount);
                    Need(r.HasLeft && r.HasRight, "left and right turns", 0f);
                    break;
                case TrackProfile.Speedway:
                    Need(r.LongestStraight >= 250f, "longest straight >= 250", r.LongestStraight);
                    Need(r.MinRadius >= 60f, "min radius >= 60", r.MinRadius);
                    break;
                case TrackProfile.Procedural:
                    // character floor so the generator does not only emit convex ovals
                    Need(r.HasLeft && r.HasRight, "left and right turns", 0f);
                    Need(r.MinRadius <= ProceduralMaxMinRadius, "a corner with R <= 40", r.MinRadius);
                    break;
                case TrackProfile.Elevation:
                    break; // Elevation rules: M6 step 3
            }
            return fail;
        }

        public static Report Validate(TrackGeometry g)
        {
            var r = new Report { Length = g.Length, RequiredSeparation = 2f * g.HalfWidth + 4f };
            int n = g.SampleCount;
            float ds = g.SampleSpacing;

            r.MinRadius = float.MaxValue;
            double totalTurn = 0.0;
            for (int k = 0; k < n; k++)
            {
                totalTurn += g.SampleTurnSign(k) * (double)ds / g.SampleRadius(k);
                r.MinRadius = Mathf.Min(r.MinRadius, g.SampleRadius(k));
                Vector3 p = g.SamplePoint(k);
                r.MaxAbsCoordinate = Mathf.Max(r.MaxAbsCoordinate, Mathf.Max(Mathf.Abs(p.x), Mathf.Abs(p.z)));
            }
            r.TotalTurnDeg = (float)(totalTurn * Mathf.Rad2Deg);

            // corners: contiguous runs with radius < CornerRadius, starting from a non-corner sample
            int start = 0;
            while (start < n && g.SampleRadius(start) < CornerRadius) start++;
            if (start == n) start = 0;
            Corner? cur = null;
            float turn = 0f;
            for (int i = 0; i <= n; i++)
            {
                int k = (start + i) % n;
                bool inside = i < n && g.SampleRadius(k) < CornerRadius;
                if (inside)
                {
                    if (cur == null) { cur = new Corner { StartS = g.SampleS(k), MinRadius = float.MaxValue }; turn = 0f; }
                    var c = cur.Value;
                    c.MinRadius = Mathf.Min(c.MinRadius, g.SampleRadius(k));
                    c.EndS = g.SampleS(k);
                    cur = c;
                    turn += g.SampleTurnSign(k) * ds / g.SampleRadius(k);
                }
                else if (cur != null)
                {
                    var c = cur.Value;
                    c.Sign = turn > 0f ? 1 : -1;
                    c.TurnDeg = Mathf.Abs(turn) * Mathf.Rad2Deg;
                    r.Corners.Add(c);
                    cur = null;
                }
            }

            foreach (var c in r.Corners)
            {
                if (c.MinRadius >= 12f && c.MinRadius <= 20f && c.TurnDeg >= 120f) { r.HasHairpin = true; r.HairpinCount++; }
                if (c.MinRadius >= 60f && c.MinRadius <= 80f) r.HasFastCorner = true;
                if (c.Sign > 0 && c.TurnDeg > 30f) r.HasLeft = true;
                if (c.Sign < 0 && c.TurnDeg > 30f) r.HasRight = true;
            }
            var sorted = new List<Corner>(r.Corners);
            sorted.Sort((a, b) => a.StartS.CompareTo(b.StartS));
            for (int i = 0; i < sorted.Count; i++)
            {
                var a = sorted[i];
                var b = sorted[(i + 1) % sorted.Count];
                float gap = b.StartS - a.EndS;
                if (gap < 0f) gap += g.Length;
                if (a.Sign != b.Sign && gap < 30f && a.MinRadius < 60f && b.MinRadius < 60f) { r.HasChicane = true; r.ChicaneCount++; }
            }

            // longest straight: run with radius > threshold
            int best = 0, run = 0;
            for (int i = 0; i < 2 * n; i++)
            {
                if (g.SampleRadius(i % n) > StraightThresholdRadius) { run++; best = Math.Max(best, run); }
                else run = 0;
            }
            r.LongestStraight = Math.Min(best, n) * ds;

            // self-proximity: samples further apart than SeparationArc along the track must stay apart
            int arcSamples = (int)Math.Ceiling(SeparationArc / ds);
            float minSep2 = float.MaxValue;
            for (int a = 0; a < n; a += 2)
            {
                Vector3 pa = g.SamplePoint(a);
                for (int b = a + 1; b < n; b += 2)
                {
                    int d = Math.Min(b - a, n - (b - a));
                    if (d <= arcSamples) continue;
                    float d2 = (g.SamplePoint(b) - pa).sqrMagnitude;
                    if (d2 < minSep2) minSep2 = d2;
                }
            }
            r.MinSeparation = Mathf.Sqrt(minSep2);
            return r;
        }
    }
}
