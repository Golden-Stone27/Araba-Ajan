using System;
using System.Globalization;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>
    /// Seeded procedural tracks "proc:&lt;seed&gt;" (M6, contracts C0.20). PCG32 (DeterministicRng, own stream) drives
    /// an accept/reject loop: K knots on a randomly stretched ellipse with jittered angles and radial perturbation,
    /// a random-strength smoothing pass, random rotation and mirroring (clockwise tracks too), random width, then an exact rescale
    /// to a random target length (centripetal Catmull-Rom is scale-covariant) and TrackValidator's universal rules plus
    /// the Procedural profile (left and right corners, at least one tight corner).
    /// The first candidate that passes is returned, so the same seed always yields the same track; control points
    /// are rounded to centimetres. Changing anything here changes proc tracks: bump Version and say so in C0.20.
    /// </summary>
    public static class ProceduralTrackGenerator
    {
        public const string Prefix = "proc:";
        public const int Version = 1;
        public const int MaxAttempts = 500;
        public const ulong RngStream = 0x70726f63UL; // "proc"

        public sealed class Result
        {
            public TrackDefinition Definition;
            public TrackValidator.Report Report;
            public int Attempts;
        }

        public static string Name(long seed) => Prefix + seed.ToString(CultureInfo.InvariantCulture);

        /// <summary>"proc:&lt;seed&gt;" with a non-negative decimal int64 seed (no sign, no spaces).</summary>
        public static bool TryParseName(string name, out long seed)
        {
            seed = 0;
            if (name == null || !name.StartsWith(Prefix, StringComparison.Ordinal)) return false;
            string digits = name.Substring(Prefix.Length);
            if (digits.Length == 0 || digits.Length > 19) return false;
            foreach (char ch in digits) if (ch < '0' || ch > '9') return false;
            return long.TryParse(digits, NumberStyles.None, CultureInfo.InvariantCulture, out seed);
        }

        public static Result Generate(long seed)
        {
            if (seed < 0) throw new ArgumentOutOfRangeException(nameof(seed), "procedural seeds are non-negative");
            var rng = new DeterministicRng(seed, RngStream);
            for (int attempt = 1; attempt <= MaxAttempts; attempt++)
            {
                Vector2[] pts = Candidate(rng, out float width, out double targetLength);
                TrackDefinition def = Make(seed, attempt, pts, width);
                var g = new TrackGeometry(def);
                double k = targetLength / g.Length;
                for (int i = 0; i < pts.Length; i++)
                    pts[i] = new Vector2((float)Math.Round(pts[i].x * k, 2), (float)Math.Round(pts[i].y * k, 2));
                def.controlPoints = pts;
                g = new TrackGeometry(def);
                TrackValidator.Report r = TrackValidator.Validate(g);
                if (TrackValidator.CheckProfile(def, r).Count == 0)
                    return new Result { Definition = def, Report = r, Attempts = attempt };
                UnityEngine.Object.DestroyImmediate(def);
            }
            throw new InvalidOperationException("no valid procedural track for seed " + seed + " in " + MaxAttempts + " attempts");
        }

        static Vector2[] Candidate(DeterministicRng rng, out float width, out double targetLength)
        {
            int knots = 10 + (int)(rng.NextUInt() % 9u);         // 10..18
            double aspect = rng.Range(1f, 1.7f);                 // ellipse semi-axes (aspect, 1)
            double rotation = rng.Range(0f, 2f * Mathf.PI);
            bool mirror = (rng.NextUInt() & 1u) == 1u;           // clockwise track
            width = 10.5f + 0.5f * (rng.NextUInt() % 7u);         // 10.5..13.5 m
            targetLength = rng.Range(800f, 1400f);

            var theta = new double[knots];
            var radius = new double[knots];
            for (int i = 0; i < knots; i++)
            {
                theta[i] = 2.0 * Math.PI * (i + rng.Range(-0.3f, 0.3f)) / knots;
                double c = Math.Cos(theta[i]), s = Math.Sin(theta[i]);
                double ellipse = aspect / Math.Sqrt(c * c + aspect * aspect * s * s);
                radius[i] = ellipse * rng.Range(0.45f, 1.25f);
            }
            double blend = rng.Range(0f, 0.5f);                  // 0 = raw knots, 0.5 = (1/4, 1/2, 1/4) smoothing
            var smooth = new double[knots];
            for (int i = 0; i < knots; i++)
                smooth[i] = (1.0 - blend) * radius[i] + 0.5 * blend * (radius[(i - 1 + knots) % knots] + radius[(i + 1) % knots]);

            var pts = new Vector2[knots];
            const double scale = 150.0; // rough size; the exact length comes from the rescale
            double cr = Math.Cos(rotation), sr = Math.Sin(rotation);
            for (int i = 0; i < knots; i++)
            {
                double x = scale * smooth[i] * Math.Cos(theta[i]), z = scale * smooth[i] * Math.Sin(theta[i]);
                double rx = cr * x - sr * z, rz = sr * x + cr * z;
                pts[i] = new Vector2((float)rx, (float)(mirror ? -rz : rz));
            }
            return pts;
        }

        static TrackDefinition Make(long seed, int attempt, Vector2[] pts, float width)
        {
            TrackDefinition def = TrackDefinition.CreateDefault();
            def.hideFlags = HideFlags.DontSave;
            def.name = Name(seed);
            def.trackId = Name(seed);
            def.profile = TrackProfile.Procedural;
            def.width = width;
            def.controlPoints = pts;
            def.notes = string.Format(CultureInfo.InvariantCulture, "ProceduralTrackGenerator v{0} seed {1} attempt {2}", Version, seed, attempt);
            return def;
        }
    }
}
