using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>
    /// Readable track source: a closed chain of straights and constant-radius arcs, baked into Catmull-Rom control
    /// points (x, z) for a TrackDefinition. The start/finish line (s = 0) is the start of the first segment.
    /// Heading 0 = +x; positive arc angles turn left (counter-clockwise from above, TrackGeometry turn sign +1).
    /// Closure: the layout must turn exactly ±360°. The remaining position gap is closed by lengthening/shortening
    /// the two straights marked flex (2×2 solve); any tiny residual is then distributed along the arc length.
    /// Straights are marked flex only if they are not parallel. Everything runs in double precision.
    /// </summary>
    public sealed class TrackLayout
    {
        public const float DefaultPointSpacing = 6f;
        public const double MaxHeadingErrorDeg = 1e-3;
        public const double MaxResidual = 0.5; // m, without flex straights

        public struct Segment
        {
            public double Length;   // straights
            public double Radius;   // arcs (0 = straight)
            public double AngleDeg; // arcs, + = left
            public bool Flex;
            public bool IsArc => Radius > 0.0;
            public double ArcLength => IsArc ? Radius * Math.Abs(AngleDeg) * Math.PI / 180.0 : Length;
        }

        public sealed class BakeResult
        {
            public Vector2[] ControlPoints;
            public double FlexDelta0, FlexDelta1; // metres added to the flex straights
            public double Residual;               // gap distributed after the flex solve
            public double NominalLength;          // polyline length of the segments (after flex)

            public override string ToString() => string.Format(CultureInfo.InvariantCulture,
                "points={0} flex=({1:F2}, {2:F2}) m residual={3:E2} m L_nominal={4:F1} m",
                ControlPoints.Length, FlexDelta0, FlexDelta1, Residual, NominalLength);
        }

        readonly List<Segment> _segments = new List<Segment>();
        public IReadOnlyList<Segment> Segments => _segments;
        public float PointSpacing { get; set; } = DefaultPointSpacing;

        public TrackLayout Straight(double length, bool flex = false)
        {
            if (!(length > 0.0)) throw new ArgumentOutOfRangeException(nameof(length));
            _segments.Add(new Segment { Length = length, Flex = flex });
            return this;
        }

        /// <param name="angleDeg">signed turn, + = left</param>
        public TrackLayout Arc(double radius, double angleDeg)
        {
            if (!(radius > 0.0) || angleDeg == 0.0) throw new ArgumentOutOfRangeException(nameof(radius));
            _segments.Add(new Segment { Radius = radius, AngleDeg = angleDeg });
            return this;
        }

        public TrackLayout Left(double radius, double angleDeg) => Arc(radius, Math.Abs(angleDeg));
        public TrackLayout Right(double radius, double angleDeg) => Arc(radius, -Math.Abs(angleDeg));

        public double TotalTurnDeg
        {
            get
            {
                double t = 0.0;
                foreach (Segment s in _segments) if (s.IsArc) t += s.AngleDeg;
                return t;
            }
        }

        /// <summary>Bakes control points, centred on the origin (bounding box) and rounded to centimetres.</summary>
        public BakeResult Bake()
        {
            if (_segments.Count == 0 || _segments[0].IsArc) throw new InvalidOperationException("layout must start with a straight (start/finish line)");
            double turn = TotalTurnDeg;
            if (Math.Abs(Math.Abs(turn) - 360.0) > MaxHeadingErrorDeg)
                throw new InvalidOperationException(string.Format(CultureInfo.InvariantCulture, "layout turns {0:F4} deg, expected +-360", turn));

            var segs = new List<Segment>(_segments);
            var result = new BakeResult();

            // flex solve: gap E = end - start; add d0·u0 + d1·u1 = -E
            var flex = new List<int>();
            for (int i = 0; i < segs.Count; i++) if (segs[i].Flex) flex.Add(i);
            if (flex.Count != 0 && flex.Count != 2) throw new InvalidOperationException("mark exactly 0 or 2 straights as flex");
            Walk(segs, null, out double ex, out double ez, out double[] headings);
            if (flex.Count == 2)
            {
                double h0 = headings[flex[0]], h1 = headings[flex[1]];
                double ax = Math.Cos(h0), az = Math.Sin(h0), bx = Math.Cos(h1), bz = Math.Sin(h1);
                double det = ax * bz - az * bx;
                if (Math.Abs(det) < 0.2) throw new InvalidOperationException("flex straights are (nearly) parallel");
                double d0 = (-ex * bz + ez * bx) / det;
                double d1 = (-az * -ex + ax * -ez) / det;
                result.FlexDelta0 = d0;
                result.FlexDelta1 = d1;
                Segment s0 = segs[flex[0]], s1 = segs[flex[1]];
                s0.Length += d0;
                s1.Length += d1;
                if (s0.Length < 1.0 || s1.Length < 1.0) throw new InvalidOperationException("flex solve made a straight vanish");
                segs[flex[0]] = s0;
                segs[flex[1]] = s1;
                Walk(segs, null, out ex, out ez, out _);
            }
            result.Residual = Math.Sqrt(ex * ex + ez * ez);
            if (flex.Count == 0 && result.Residual > MaxResidual)
                throw new InvalidOperationException(string.Format(CultureInfo.InvariantCulture, "layout does not close: gap {0:F3} m", result.Residual));

            var pts = new List<(double x, double z, double s)>();
            Walk(segs, pts, out _, out _, out _);
            double total = 0.0;
            foreach (Segment s in segs) total += s.ArcLength;
            result.NominalLength = total;

            double minX = double.MaxValue, maxX = double.MinValue, minZ = double.MaxValue, maxZ = double.MinValue;
            var xs = new double[pts.Count];
            var zs = new double[pts.Count];
            for (int i = 0; i < pts.Count; i++)
            {
                double f = pts[i].s / total; // distribute the residual gap linearly along the arc length
                xs[i] = pts[i].x - ex * f;
                zs[i] = pts[i].z - ez * f;
                minX = Math.Min(minX, xs[i]); maxX = Math.Max(maxX, xs[i]);
                minZ = Math.Min(minZ, zs[i]); maxZ = Math.Max(maxZ, zs[i]);
            }
            double cx = 0.5 * (minX + maxX), cz = 0.5 * (minZ + maxZ);
            result.ControlPoints = new Vector2[pts.Count];
            for (int i = 0; i < pts.Count; i++)
                result.ControlPoints[i] = new Vector2((float)Math.Round(xs[i] - cx, 2), (float)Math.Round(zs[i] - cz, 2));
            return result;
        }

        /// <summary>Walks the chain from (0, 0) heading +x. Emits control points (segment starts + interior samples).</summary>
        void Walk(List<Segment> segs, List<(double x, double z, double s)> emit, out double endX, out double endZ, out double[] headings)
        {
            double x = 0.0, z = 0.0, h = 0.0, s = 0.0;
            headings = new double[segs.Count];
            for (int i = 0; i < segs.Count; i++)
            {
                Segment seg = segs[i];
                headings[i] = h;
                double len = seg.ArcLength;
                int n = Math.Max(1, (int)Math.Round(len / PointSpacing));
                if (!seg.IsArc)
                {
                    double dx = Math.Cos(h), dz = Math.Sin(h);
                    if (emit != null) for (int k = 0; k < n; k++) emit.Add((x + dx * len * k / n, z + dz * len * k / n, s + len * k / n));
                    x += dx * len;
                    z += dz * len;
                }
                else
                {
                    double sign = Math.Sign(seg.AngleDeg);
                    double sweep = seg.AngleDeg * Math.PI / 180.0;
                    // centre is to the left (+90°) for left turns, to the right for right turns
                    double cx = x + seg.Radius * Math.Cos(h + sign * Math.PI / 2), cz = z + seg.Radius * Math.Sin(h + sign * Math.PI / 2);
                    double a0 = Math.Atan2(z - cz, x - cx);
                    if (emit != null)
                        for (int k = 0; k < n; k++)
                        {
                            double a = a0 + sweep * k / n;
                            emit.Add((cx + seg.Radius * Math.Cos(a), cz + seg.Radius * Math.Sin(a), s + len * k / n));
                        }
                    double a1 = a0 + sweep;
                    x = cx + seg.Radius * Math.Cos(a1);
                    z = cz + seg.Radius * Math.Sin(a1);
                    h += sweep;
                }
                s += len;
            }
            endX = x;
            endZ = z;
        }
    }
}
