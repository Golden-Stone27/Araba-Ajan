using System;
using System.Collections.Generic;
using UnityEngine;

namespace Racing.Core
{
    public interface ITrack
    {
        float Length { get; }
        float HalfWidth { get; }
        int CheckpointCount { get; }
        Checkpoint GetCheckpoint(int index);
        /// <summary>Projects onto the centreline. sampleHint &lt; 0 forces a full scan.</summary>
        TrackProjection Project(Vector3 worldPos, int sampleHint);
        Vector3 TangentAt(float s);
        Vector3 PointAt(float s);
        Vector3 RightAt(float s);
        float RadiusAt(float s);
        Pose SpawnPose(float s, float lateral, float headingOffsetDeg);
    }

    public static class CatmullRom
    {
        /// <summary>Closed centripetal (alpha = 0.5) Catmull-Rom, dense polyline without the closing duplicate.</summary>
        public static List<Vector2> SampleClosed(IReadOnlyList<Vector2> pts, float alpha, int perSegment)
        {
            int n = pts.Count;
            var output = new List<Vector2>(n * perSegment);
            for (int i = 0; i < n; i++)
            {
                Vector2 p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n];
                double t0 = 0.0;
                double t1 = Knot(t0, p0, p1, alpha);
                double t2 = Knot(t1, p1, p2, alpha);
                double t3 = Knot(t2, p2, p3, alpha);
                for (int k = 0; k < perSegment; k++)
                {
                    double t = t1 + (t2 - t1) * k / perSegment;
                    Vector2d a1 = Lerp(p0, p1, t0, t1, t), a2 = Lerp(p1, p2, t1, t2, t), a3 = Lerp(p2, p3, t2, t3, t);
                    Vector2d b1 = Lerp(a1, a2, t0, t2, t), b2 = Lerp(a2, a3, t1, t3, t);
                    Vector2d c = Lerp(b1, b2, t1, t2, t);
                    output.Add(new Vector2((float)c.X, (float)c.Y));
                }
            }
            return output;
        }

        static double Knot(double ti, Vector2 a, Vector2 b, float alpha)
        {
            double d = Math.Sqrt((double)(b.x - a.x) * (b.x - a.x) + (double)(b.y - a.y) * (b.y - a.y));
            return ti + Math.Pow(Math.Max(d, 1e-6), alpha);
        }

        readonly struct Vector2d
        {
            public readonly double X, Y;
            public Vector2d(double x, double y) { X = x; Y = y; }
            public static implicit operator Vector2d(Vector2 v) => new Vector2d(v.x, v.y);
        }

        static Vector2d Lerp(Vector2d a, Vector2d b, double ta, double tb, double t)
        {
            double wa = (tb - t) / (tb - ta), wb = (t - ta) / (tb - ta);
            return new Vector2d(wa * a.X + wb * b.X, wa * a.Y + wb * b.Y);
        }
    }

    /// <summary>
    /// Pure (GameObject-free) track geometry: arc-length resampled centreline, tangents, right vectors,
    /// signed curvature radii and checkpoint gates. s is the horizontal (XZ) arc length. Samples carry the
    /// definition's elevation (y = HeightAt(s / L), 0 for flat tracks); tangents are 3D (along the slope),
    /// right vectors horizontal, radii and projection horizontal (M6, C0.20: bit-identical for y = 0).
    /// </summary>
    public sealed class TrackGeometry : ITrack
    {
        public const int ProjectionWindow = 20;
        const int RadiusHalfSpan = 4;
        public const float StraightRadius = 1e4f;

        readonly Vector3[] _p;
        readonly Vector3[] _t;
        readonly Vector3[] _n;
        readonly float[] _radius;      // unsigned, capped at StraightRadius
        readonly int[] _turnSign;      // +1 left, -1 right, 0 straight
        readonly Checkpoint[] _gates;

        public float Length { get; }
        public float SampleSpacing { get; }
        public float HalfWidth { get; }
        public int SampleCount => _p.Length;
        public int CheckpointCount => _gates.Length;
        public TrackDefinition Definition { get; }

        public TrackGeometry(TrackDefinition def)
        {
            Definition = def;
            HalfWidth = def.width * 0.5f;
            List<Vector2> dense = CatmullRom.SampleClosed(def.controlPoints, def.catmullAlpha, 200);

            // arc-length resample to an exact uniform spacing close to def.sampleSpacing
            int m = dense.Count;
            var cum = new double[m + 1];
            for (int i = 1; i <= m; i++) cum[i] = cum[i - 1] + Vector2.Distance(dense[i - 1], dense[i % m]);
            double total = cum[m];
            int count = Math.Max(16, (int)Math.Floor(total / def.sampleSpacing));
            double ds = total / count;
            Length = (float)total;
            SampleSpacing = (float)ds;

            _p = new Vector3[count];
            int j = 0;
            for (int k = 0; k < count; k++)
            {
                double s = k * ds;
                while (cum[j + 1] < s) j++;
                double u = (s - cum[j]) / (cum[j + 1] - cum[j]);
                Vector2 a = dense[j], b = dense[(j + 1) % m];
                _p[k] = new Vector3((float)(a.x + u * (b.x - a.x)), 0f, (float)(a.y + u * (b.y - a.y)));
            }
            if (def.HasElevation)
                for (int k = 0; k < count; k++) _p[k].y = (float)def.HeightAt(k * ds / total);

            _t = new Vector3[count];
            _n = new Vector3[count];
            _radius = new float[count];
            _turnSign = new int[count];
            for (int k = 0; k < count; k++)
            {
                Vector3 d = _p[(k + 1) % count] - _p[(k - 1 + count) % count];
                _t[k] = d.normalized;
                _n[k] = Vector3.Cross(Vector3.up, _t[k]).normalized;

                Vector3 a = Flat(_p[(k - RadiusHalfSpan + count) % count]), b = Flat(_p[k]), c = Flat(_p[(k + RadiusHalfSpan) % count]);
                float cross = (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x);
                float area2 = Mathf.Abs(cross);
                if (area2 < 1e-6f)
                {
                    _radius[k] = StraightRadius;
                    _turnSign[k] = 0;
                }
                else
                {
                    float r = Vector3.Distance(a, b) * Vector3.Distance(b, c) * Vector3.Distance(c, a) / (2f * area2);
                    _radius[k] = Mathf.Min(r, StraightRadius);
                    _turnSign[k] = cross > 0f ? 1 : -1; // +: counter-clockwise from above = left turn
                }
            }

            int gateCount = Math.Max(3, (int)Math.Floor(Length / def.checkpointSpacing));
            float gateSpacing = Length / gateCount;
            _gates = new Checkpoint[gateCount];
            for (int i = 0; i < gateCount; i++)
            {
                float s = i * gateSpacing;
                _gates[i] = new Checkpoint(i, PointAt(s), TangentAt(s), RightAt(s), HalfWidth, s);
            }
        }

        public Vector3 SamplePoint(int k) => _p[Wrap(k)];
        public Vector3 SampleTangent(int k) => _t[Wrap(k)];
        public Vector3 SampleRight(int k) => _n[Wrap(k)];
        public float SampleRadius(int k) => _radius[Wrap(k)];
        public int SampleTurnSign(int k) => _turnSign[Wrap(k)];
        public float SampleS(int k) => Wrap(k) * SampleSpacing;

        int Wrap(int k) { int n = _p.Length; k %= n; return k < 0 ? k + n : k; }

        static Vector3 Flat(Vector3 v) => new Vector3(v.x, 0f, v.z);

        public float WrapS(float s)
        {
            s %= Length;
            return s < 0f ? s + Length : s;
        }

        void Locate(float s, out int k0, out int k1, out float u)
        {
            float x = WrapS(s) / SampleSpacing;
            k0 = (int)Math.Floor(x);
            u = x - k0;
            k0 = Wrap(k0);
            k1 = Wrap(k0 + 1);
        }

        public Vector3 PointAt(float s)
        {
            Locate(s, out int a, out int b, out float u);
            return Vector3.LerpUnclamped(_p[a], _p[b], u);
        }

        public Vector3 TangentAt(float s)
        {
            Locate(s, out int a, out int b, out float u);
            return Vector3.LerpUnclamped(_t[a], _t[b], u).normalized;
        }

        public Vector3 RightAt(float s) => Vector3.Cross(Vector3.up, TangentAt(s)).normalized;

        public float RadiusAt(float s)
        {
            Locate(s, out int a, out int b, out float u);
            return Mathf.Lerp(_radius[a], _radius[b], u);
        }

        public Checkpoint GetCheckpoint(int index)
        {
            int n = _gates.Length;
            index %= n;
            return _gates[index < 0 ? index + n : index];
        }

        /// <summary>Index of the first gate strictly ahead of arc position s (wraps to 0).</summary>
        public int NextGateAfter(float s)
        {
            s = WrapS(s);
            for (int i = 0; i < _gates.Length; i++)
                if (_gates[i].S > s) return i;
            return 0;
        }

        public TrackProjection Project(Vector3 worldPos, int sampleHint)
        {
            int n = _p.Length;
            int start, span;
            if (sampleHint < 0) { start = 0; span = n; }
            else { start = sampleHint - ProjectionWindow; span = 2 * ProjectionWindow + 1; }

            float best = float.MaxValue;
            int bestK = 0;
            float bestU = 0f;
            Vector3 q = new Vector3(worldPos.x, 0f, worldPos.z);
            for (int i = 0; i < span; i++)
            {
                int k = Wrap(start + i);
                Vector3 a = Flat(_p[k]), b = Flat(_p[Wrap(k + 1)]);
                Vector3 ab = b - a;
                float u = Mathf.Clamp01(Vector3.Dot(q - a, ab) / Mathf.Max(ab.sqrMagnitude, 1e-9f));
                float d2 = (a + ab * u - q).sqrMagnitude;
                if (d2 < best) { best = d2; bestK = k; bestU = u; }
            }

            Vector3 pa = _p[bestK], pb = _p[Wrap(bestK + 1)];
            Vector3 dir = (pb - pa).normalized; // along the slope (reward v·t̂)
            Vector3 fa = Flat(pa), fb = Flat(pb);
            Vector3 foot = fa + (fb - fa) * bestU;
            Vector3 right = Vector3.Cross(Vector3.up, (fb - fa).normalized);
            float lateral = Vector3.Dot(q - foot, right);
            float s = (bestK + bestU) * SampleSpacing;
            return new TrackProjection(WrapS(s), lateral, dir, bestK);
        }

        public Pose SpawnPose(float s, float lateral, float headingOffsetDeg)
        {
            Vector3 t = TangentAt(s);
            Vector3 pos = PointAt(s) + RightAt(s) * lateral;
            Quaternion rot = Quaternion.LookRotation(t, Vector3.up) * Quaternion.Euler(0f, headingOffsetDeg, 0f);
            return new Pose(pos, rot);
        }
    }
}
