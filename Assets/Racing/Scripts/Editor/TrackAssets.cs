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

        /// <summary>Catalog order is append-only: published indices never move.</summary>
        public static readonly string[] CatalogOrder = { ProjectSetup.TrackDefinitionPath };

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
