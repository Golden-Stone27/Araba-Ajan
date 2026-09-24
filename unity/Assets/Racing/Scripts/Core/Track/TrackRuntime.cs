using System.Collections.Generic;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>
    /// Builds the physical track from a TrackDefinition: one ground box (Road layer), a chain of
    /// 1 m thick wall BoxColliders per side (Wall layer, collider-only) and visual meshes (no colliders).
    /// No trigger colliders are created (C0.4).
    /// M6 (C0.20): RoadColliderMode.MeshStrip adds a MeshCollider road strip that follows the elevation (Road layer,
    /// under the walls) and drops the ground box below it; wall colliders can extend invisibly below/above the road
    /// so the horizontal ray sensor keeps seeing them on grades. Defaults build exactly the M1 track.
    /// </summary>
    public sealed class TrackRuntime : MonoBehaviour
    {
        public TrackGeometry Geometry { get; private set; }
        public int WallColliderCount { get; private set; }

        [SerializeField] bool drawGizmos = true;

        public void Build(TrackDefinition def, Material road = null, Material wall = null, Material ground = null)
        {
            for (int i = transform.childCount - 1; i >= 0; i--) DestroyImmediate(transform.GetChild(i).gameObject);
            Geometry = new TrackGeometry(def);

            BuildGround(ground ?? MakeMaterial(new Color(0.25f, 0.45f, 0.22f)), def.roadCollider == RoadColliderMode.MeshStrip);
            if (def.roadCollider == RoadColliderMode.MeshStrip) BuildRoadCollider(def);
            BuildRoadMesh(road ?? MakeMaterial(new Color(0.18f, 0.18f, 0.2f)));
            var wallMat = wall ?? MakeMaterial(new Color(0.8f, 0.15f, 0.15f));
            float offset = Geometry.HalfWidth + def.wallThickness * 0.5f;
            BuildWallSide("WallRight", +offset, def, wallMat);
            BuildWallSide("WallLeft", -offset, def, wallMat);
            BuildStartLine();
            Physics.SyncTransforms();
        }

        /// <summary>GroundBox mode: the ground cube's top (y = 0) is the road. MeshStrip mode: it sits 0.1 m lower (visual only).</summary>
        void BuildGround(Material mat, bool belowStrip)
        {
            float extent = 0f;
            for (int k = 0; k < Geometry.SampleCount; k++)
            {
                Vector3 p = Geometry.SamplePoint(k);
                extent = Mathf.Max(extent, Mathf.Max(Mathf.Abs(p.x), Mathf.Abs(p.z)));
            }
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = "Ground";
            go.layer = RacingLayers.Road;
            go.transform.SetParent(transform, false);
            float size = 2f * extent + 200f;
            go.transform.localPosition = new Vector3(0f, belowStrip ? -0.6f : -0.5f, 0f);
            go.transform.localScale = new Vector3(size, 1f, size);
            go.GetComponent<Renderer>().sharedMaterial = mat;
            go.isStatic = true;
        }

        void BuildRoadMesh(Material mat)
        {
            int n = Geometry.SampleCount;
            var verts = new Vector3[2 * (n + 1)];
            var uvs = new Vector2[verts.Length];
            var tris = new int[6 * n];
            for (int k = 0; k <= n; k++)
            {
                Vector3 p = Geometry.SamplePoint(k), r = Geometry.SampleRight(k) * Geometry.HalfWidth;
                verts[2 * k] = p - r + Vector3.up * 0.01f;
                verts[2 * k + 1] = p + r + Vector3.up * 0.01f;
                float v = k * Geometry.SampleSpacing / 10f;
                uvs[2 * k] = new Vector2(0f, v);
                uvs[2 * k + 1] = new Vector2(1f, v);
            }
            for (int k = 0; k < n; k++)
            {
                int a = 2 * k, t = 6 * k;
                tris[t] = a; tris[t + 1] = a + 2; tris[t + 2] = a + 1;
                tris[t + 3] = a + 1; tris[t + 4] = a + 2; tris[t + 5] = a + 3;
            }
            MakeMeshObject("RoadVisual", verts, uvs, tris, mat);
        }

        /// <summary>Road-layer MeshCollider strip, half-width W/2 + wallThickness + margin, flat cross-section, up-facing.</summary>
        void BuildRoadCollider(TrackDefinition def)
        {
            int n = Geometry.SampleCount;
            float half = Geometry.HalfWidth + def.wallThickness + def.roadColliderMargin;
            var verts = new Vector3[2 * (n + 1)];
            var tris = new int[6 * n];
            for (int k = 0; k <= n; k++)
            {
                Vector3 p = Geometry.SamplePoint(k), r = Geometry.SampleRight(k) * half;
                verts[2 * k] = p - r;
                verts[2 * k + 1] = p + r;
            }
            for (int k = 0; k < n; k++)
            {
                int a = 2 * k, t = 6 * k;
                tris[t] = a; tris[t + 1] = a + 2; tris[t + 2] = a + 1;
                tris[t + 3] = a + 1; tris[t + 4] = a + 2; tris[t + 5] = a + 3;
            }
            var mesh = new Mesh { name = "RoadCollider", indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            mesh.vertices = verts;
            mesh.triangles = tris;
            mesh.RecalculateBounds();
            var go = new GameObject("RoadCollider") { layer = RacingLayers.Road, isStatic = true };
            go.transform.SetParent(transform, false);
            go.AddComponent<MeshCollider>().sharedMesh = mesh;
        }

        void BuildWallSide(string name, float offset, TrackDefinition def, Material mat)
        {
            var root = new GameObject(name);
            root.transform.SetParent(transform, false);
            root.isStatic = true;

            // offset curve points every ~wallSegmentLength along the centreline
            int n = Geometry.SampleCount;
            int stride = Mathf.Max(1, Mathf.RoundToInt(def.wallSegmentLength / Geometry.SampleSpacing));
            var q = new List<Vector3>();
            for (int k = 0; k < n; k += stride) q.Add(Geometry.SamplePoint(k) + Geometry.SampleRight(k) * offset);

            int count = q.Count;
            for (int i = 0; i < count; i++)
            {
                Vector3 a = q[i], b = q[(i + 1) % count];
                Vector3 d = b - a;
                float len = d.magnitude;
                if (len < 1e-3f) continue;
                var go = new GameObject("W" + i);
                go.layer = RacingLayers.Wall;
                go.isStatic = true;
                go.transform.SetParent(root.transform, false);
                // collider spans [road − extraBelow, road + wallHeight + extraAbove]; the defaults give the M1 box exactly
                float colliderHeight = def.wallHeight + def.wallColliderExtraAbove + def.wallColliderExtraBelow;
                float centreY = (def.wallHeight + def.wallColliderExtraAbove - def.wallColliderExtraBelow) * 0.5f;
                go.transform.SetPositionAndRotation((a + b) * 0.5f + Vector3.up * centreY,
                    Quaternion.LookRotation(d / len, Vector3.up));
                var box = go.AddComponent<BoxCollider>();
                box.size = new Vector3(def.wallThickness, colliderHeight, len * 1.1f);
                WallColliderCount++;
            }

            // visual: extruded strip (inner face, top, outer face) following the offset curve
            float half = def.wallThickness * 0.5f;
            var verts = new List<Vector3>();
            var tris = new List<int>();
            var uvs = new List<Vector2>();
            for (int i = 0; i <= count; i++)
            {
                int k = (i * stride) % n;
                Vector3 c = Geometry.SamplePoint(k) + Geometry.SampleRight(k) * offset;
                Vector3 r = Geometry.SampleRight(k) * half;
                Vector3 up = Vector3.up * def.wallHeight;
                verts.Add(c - r); verts.Add(c - r + up); verts.Add(c + r + up); verts.Add(c + r);
                float v = i * def.wallSegmentLength / 4f;
                uvs.Add(new Vector2(0f, v)); uvs.Add(new Vector2(0.33f, v)); uvs.Add(new Vector2(0.66f, v)); uvs.Add(new Vector2(1f, v));
            }
            for (int i = 0; i < count; i++)
            {
                int a = 4 * i, b = 4 * (i + 1);
                for (int f = 0; f < 3; f++)
                {
                    int a0 = a + f, a1 = a + f + 1, b0 = b + f, b1 = b + f + 1;
                    tris.Add(a0); tris.Add(b0); tris.Add(a1);
                    tris.Add(a1); tris.Add(b0); tris.Add(b1);
                    tris.Add(a0); tris.Add(a1); tris.Add(b0);
                    tris.Add(a1); tris.Add(b1); tris.Add(b0);
                }
            }
            MakeMeshObject(name + "Visual", verts.ToArray(), uvs.ToArray(), tris.ToArray(), mat).transform.SetParent(root.transform, true);
        }

        void BuildStartLine()
        {
            var g = Geometry.GetCheckpoint(0);
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = "StartLine";
            DestroyImmediate(go.GetComponent<Collider>());
            go.transform.SetParent(transform, false);
            go.transform.SetPositionAndRotation(g.Position + Vector3.up * 0.015f, Quaternion.LookRotation(g.Forward, Vector3.up));
            go.transform.localScale = new Vector3(2f * g.HalfWidth, 0.01f, 0.8f);
            go.GetComponent<Renderer>().sharedMaterial = MakeMaterial(Color.white);
        }

        GameObject MakeMeshObject(string name, Vector3[] verts, Vector2[] uvs, int[] tris, Material mat)
        {
            var mesh = new Mesh { name = name, indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            mesh.vertices = verts;
            mesh.uv = uvs;
            mesh.triangles = tris;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            var go = new GameObject(name);
            go.transform.SetParent(transform, false);
            go.isStatic = true;
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            go.AddComponent<MeshRenderer>().sharedMaterial = mat;
            return go;
        }

        static Material MakeMaterial(Color c)
        {
            var shader = Shader.Find("Standard") ?? Shader.Find("Universal Render Pipeline/Lit");
            var m = new Material(shader) { color = c };
            return m;
        }

        void OnDrawGizmos()
        {
            if (!drawGizmos || Geometry == null) return;
            Gizmos.color = Color.yellow;
            int n = Geometry.SampleCount;
            for (int k = 0; k < n; k += 2) Gizmos.DrawLine(Geometry.SamplePoint(k) + Vector3.up * 0.1f, Geometry.SamplePoint(k + 2) + Vector3.up * 0.1f);
            for (int i = 0; i < Geometry.CheckpointCount; i++)
            {
                var g = Geometry.GetCheckpoint(i);
                Gizmos.color = i == 0 ? Color.white : Color.cyan;
                Vector3 up = Vector3.up * 0.3f;
                Gizmos.DrawLine(g.Position - g.Right * g.HalfWidth + up, g.Position + g.Right * g.HalfWidth + up);
            }
        }
    }
}
