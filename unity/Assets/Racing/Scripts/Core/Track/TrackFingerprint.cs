using System;
using System.Globalization;
using System.Text;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>
    /// Bit-exact fingerprints of a track (M6 golden tests, contracts C0.20). Every float is written as its IEEE-754
    /// bit pattern, so any change in geometry or in the built colliders, however small, changes the hash.
    /// </summary>
    public static class TrackFingerprint
    {
        /// <summary>
        /// Centreline arrays (points, tangents, right vectors, radii, turn signs), gates, and the interpolating
        /// queries (PointAt/TangentAt/RightAt/RadiusAt/Project/SpawnPose) at fixed arc positions.
        /// </summary>
        public static string Geometry(TrackGeometry g)
        {
            var sb = new StringBuilder(1 << 20);
            F(sb, g.Length); F(sb, g.SampleSpacing); F(sb, g.HalfWidth); I(sb, g.SampleCount); I(sb, g.CheckpointCount);
            for (int k = 0; k < g.SampleCount; k++)
            {
                V(sb, g.SamplePoint(k)); V(sb, g.SampleTangent(k)); V(sb, g.SampleRight(k));
                F(sb, g.SampleRadius(k)); I(sb, g.SampleTurnSign(k));
            }
            for (int i = 0; i < g.CheckpointCount; i++)
            {
                Checkpoint c = g.GetCheckpoint(i);
                I(sb, c.Index); V(sb, c.Position); V(sb, c.Forward); V(sb, c.Right); F(sb, c.HalfWidth); F(sb, c.S);
            }
            for (float s = 0.37f; s < g.Length; s += 7.3f)
            {
                V(sb, g.PointAt(s)); V(sb, g.TangentAt(s)); V(sb, g.RightAt(s)); F(sb, g.RadiusAt(s));
                I(sb, g.NextGateAfter(s));
                TrackProjection p = g.Project(g.PointAt(s) + g.RightAt(s) * 1.7f, -1);
                F(sb, p.S); F(sb, p.Lateral); V(sb, p.Tangent); I(sb, p.SampleIndex);
                Pose pose = g.SpawnPose(s, 0.9f, 3f);
                V(sb, pose.position); Q(sb, pose.rotation);
            }
            return ConfigHash.Sha256Hex16(sb.ToString());
        }

        /// <summary>Every collider under the built track root, in hierarchy order: name, layer, type, world pose, size.</summary>
        public static string Physical(TrackRuntime runtime)
        {
            var sb = new StringBuilder(1 << 18);
            Collider[] colliders = runtime.GetComponentsInChildren<Collider>(true);
            I(sb, colliders.Length);
            foreach (Collider c in colliders)
            {
                Transform t = c.transform;
                sb.Append(t.name).Append('|').Append(c.GetType().Name).Append('|');
                I(sb, t.gameObject.layer); I(sb, c.isTrigger ? 1 : 0); I(sb, c.enabled ? 1 : 0);
                V(sb, t.position); Q(sb, t.rotation); V(sb, t.lossyScale);
                switch (c)
                {
                    case BoxCollider b: V(sb, b.center); V(sb, b.size); break;
                    case MeshCollider m:
                        I(sb, m.convex ? 1 : 0);
                        Mesh mesh = m.sharedMesh;
                        if (mesh != null)
                        {
                            foreach (Vector3 v in mesh.vertices) V(sb, v);
                            foreach (int tri in mesh.triangles) I(sb, tri);
                        }
                        break;
                }
            }
            return ConfigHash.Sha256Hex16(sb.ToString());
        }

        static void F(StringBuilder sb, float v) =>
            sb.Append(BitConverter.SingleToInt32Bits(v).ToString("x8", CultureInfo.InvariantCulture)).Append(';');

        static void I(StringBuilder sb, int v) => sb.Append(v.ToString(CultureInfo.InvariantCulture)).Append(';');
        static void V(StringBuilder sb, Vector3 v) { F(sb, v.x); F(sb, v.y); F(sb, v.z); }
        static void Q(StringBuilder sb, Quaternion q) { F(sb, q.x); F(sb, q.y); F(sb, q.z); F(sb, q.w); }
    }
}
