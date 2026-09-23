using System.Collections.Generic;
using System.Text;
using Racing.Core;
using UnityEditor;
using UnityEngine;

namespace Racing.Editor
{
    /// <summary>M6 track tooling: catalog asset (index 0 = Track_A, contracts C0.20) and track reports.</summary>
    public static class TrackAssets
    {
        const string Root = "Assets/Racing";
        public const string CatalogPath = Root + "/Config/TrackCatalog.asset";
        public const string TrackBPath = Root + "/Config/TrackDefinition_B.asset";
        public const string TrackCPath = Root + "/Config/TrackDefinition_C.asset";

        /// <summary>Catalog order is append-only: published indices never move.</summary>
        public static readonly string[] CatalogOrder = { ProjectSetup.TrackDefinitionPath, TrackBPath, TrackCPath };

        /// <summary>Layout-sourced catalog tracks (Track_A predates TrackLayout and is never re-baked).</summary>
        public static readonly (string path, string id, TrackProfile profile, float width, System.Func<TrackLayout> layout)[] Baked =
        {
            (TrackBPath, "Track_B", TrackProfile.Technical, 11f, TrackLayouts.TechnicalB),
            (TrackCPath, "Track_C", TrackProfile.Speedway, 13f, TrackLayouts.SpeedwayC),
        };

        /// <summary>Bakes the layout tracks into their assets (other fields keep the Track_A defaults), then the catalog.</summary>
        [MenuItem("Racing/Tracks/Bake Track Assets (M6)")]
        public static void BakeTrackAssets()
        {
            foreach (var t in Baked)
            {
                TrackLayout.BakeResult b = t.layout().Bake();
                var def = AssetDatabase.LoadAssetAtPath<TrackDefinition>(t.path);
                if (def == null)
                {
                    def = TrackDefinition.CreateDefault();
                    AssetDatabase.CreateAsset(def, t.path);
                }
                def.trackId = t.id;
                def.profile = t.profile;
                def.width = t.width;
                def.controlPoints = b.ControlPoints;
                def.notes = "Baked from TrackLayouts (" + t.id + "): " + b;
                EditorUtility.SetDirty(def);
                Debug.Log("[Race] baked " + t.id + ": " + b);
            }
            AssetDatabase.SaveAssets();
            SetupCatalog();
        }

        [MenuItem("Racing/Tracks/Setup Track Catalog (M6)")]
        public static TrackCatalog SetupCatalog()
        {
            var catalog = AssetDatabase.LoadAssetAtPath<TrackCatalog>(CatalogPath);
            if (catalog == null)
            {
                catalog = ScriptableObject.CreateInstance<TrackCatalog>();
                AssetDatabase.CreateAsset(catalog, CatalogPath);
            }
            var list = new List<TrackDefinition>();
            foreach (string path in CatalogOrder)
            {
                var def = AssetDatabase.LoadAssetAtPath<TrackDefinition>(path);
                if (def == null)
                {
                    Debug.LogError("[Race] catalog: missing " + path);
                    continue;
                }
                list.Add(def);
            }
            catalog.tracks = list.ToArray();
            EditorUtility.SetDirty(catalog);
            AssetDatabase.SaveAssets();
            Debug.Log("[Race] track catalog saved: " + CatalogPath + " (" + catalog.Count + " tracks)");
            return catalog;
        }

        [MenuItem("Racing/Tracks/Report Tracks (M6)")]
        public static string ReportTracks()
        {
            var catalog = AssetDatabase.LoadAssetAtPath<TrackCatalog>(CatalogPath);
            var sb = new StringBuilder();
            for (int i = 0; catalog != null && i < catalog.Count; i++)
            {
                TrackDefinition def = catalog.Get(i);
                var g = new TrackGeometry(def);
                TrackValidator.Report r = TrackValidator.Validate(g);
                List<string> fails = TrackValidator.CheckProfile(def, r);
                sb.AppendLine($"[{i}] {def.trackId} ({def.profile}) gates={g.CheckpointCount} {r}");
                sb.AppendLine(fails.Count == 0 ? "    profile: PASS" : "    profile FAIL: " + string.Join("; ", fails));
            }
            Debug.Log("[Race] tracks:\n" + sb);
            return sb.ToString();
        }
    }
}
