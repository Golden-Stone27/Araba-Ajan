using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Racing.Core;
using UnityEditor;
using UnityEngine;

namespace Racing.Editor
{
    /// <summary>M6 track tooling: catalog asset (index 0 = Track_A, contracts C0.20), track reports and the Python catalog export.</summary>
    public static class TrackAssets
    {
        const string Root = "Assets/Racing";
        public const string CatalogPath = Root + "/Config/TrackCatalog.asset";
        public const string TrackBPath = Root + "/Config/TrackDefinition_B.asset";
        public const string TrackCPath = Root + "/Config/TrackDefinition_C.asset";
        public const string TrackDPath = Root + "/Config/TrackDefinition_D.asset";

        /// <summary>Python's copy of the catalog (racing_rl.bridge.tracks), repo-relative (paths.json track_catalog).</summary>
        public static string CatalogJsonRelPath => RepoPaths.TrackCatalogRelPath;
        public const string CatalogJsonSchema = "race-track-catalog/v1";

        /// <summary>Procedural seeds whose hashes are frozen in C0.20; exported as procedural_refs.</summary>
        public static readonly long[] ProceduralRefSeeds = { 0, 1, 7, 1000 };

        public static string CatalogJsonPath => RepoPaths.TrackCatalog;

        /// <summary>Catalog order is append-only: published indices never move.</summary>
        public static readonly string[] CatalogOrder = { ProjectSetup.TrackDefinitionPath, TrackBPath, TrackCPath, TrackDPath };

        /// <summary>Layout-sourced catalog tracks (Track_A predates TrackLayout and is never re-baked).</summary>
        public static readonly (string path, string id, TrackProfile profile, float width, System.Func<TrackLayout> layout,
            System.Action<TrackDefinition, TrackLayout.BakeResult> configure)[] Baked =
        {
            (TrackBPath, "Track_B", TrackProfile.Technical, 11f, TrackLayouts.TechnicalB, null),
            (TrackCPath, "Track_C", TrackProfile.Speedway, 13f, TrackLayouts.SpeedwayC, null),
            (TrackDPath, "Track_D", TrackProfile.Elevation, 12f, TrackLayouts.HillD, ConfigureHill),
        };

        /// <summary>Track_D: elevation keys (u = s / L_nominal), MeshStrip road collider, tall wall colliders.</summary>
        public static void ConfigureHill(TrackDefinition def, TrackLayout.BakeResult b)
        {
            var keys = new ElevationKey[TrackLayouts.HillDElevation.Length];
            for (int i = 0; i < keys.Length; i++)
                keys[i] = new ElevationKey((float)System.Math.Round(TrackLayouts.HillDElevation[i].s / b.NominalLength, 5), TrackLayouts.HillDElevation[i].y);
            def.elevation = keys;
            def.roadCollider = RoadColliderMode.MeshStrip;
            def.wallColliderExtraBelow = TrackLayouts.HillWallExtension;
            def.wallColliderExtraAbove = TrackLayouts.HillWallExtension;
        }

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
                def.elevation = new ElevationKey[0];
                def.roadCollider = RoadColliderMode.GroundBox;
                def.wallColliderExtraBelow = 0f;
                def.wallColliderExtraAbove = 0f;
                t.configure?.Invoke(def, b);
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

        /// <summary>
        /// Python's view of the catalog (M6 step 5): per track the HELLO track fields, the frozen env_config_hash and
        /// track_hash, plus the C0.20 procedural reference seeds (index -1). Deterministic text (InvariantCulture, "R"
        /// floats, LF); EditMode ExportedJson_IsUpToDate compares it with the committed file.
        /// </summary>
        public static string BuildCatalogJson()
        {
            var catalog = AssetDatabase.LoadAssetAtPath<TrackCatalog>(CatalogPath);
            var sim = AssetDatabase.LoadAssetAtPath<SimConfig>(ProjectSetup.SimConfigPath);
            var vehicle = AssetDatabase.LoadAssetAtPath<VehicleConfig>(ProjectSetup.VehicleConfigPath);
            var reward = AssetDatabase.LoadAssetAtPath<RewardConfig>(MlaTools.RewardConfigPath);
            if (catalog == null || sim == null || vehicle == null || reward == null)
                throw new InvalidOperationException("catalog export: missing TrackCatalog, SimConfig, VehicleConfig or RewardConfig asset");

            var c = CultureInfo.InvariantCulture;
            var sb = new StringBuilder();
            sb.Append("{\n");
            sb.Append("  \"schema\": \"").Append(CatalogJsonSchema).Append("\",\n");
            sb.Append("  \"generated_by\": \"Racing/Tracks/Export Track Catalog (M6)\",\n");
            sb.Append("  \"obs_layout_hash\": \"").Append(ObservationSpec.LayoutHash).Append("\",\n");
            sb.Append("  \"procedural_generator_version\": ").Append(ProceduralTrackGenerator.Version.ToString(c)).Append(",\n");
            sb.Append("  \"tracks\": [\n");
            for (int i = 0; i < catalog.Count; i++)
            {
                TrackDefinition def = catalog.Get(i);
                AppendEntry(sb, def, i, def.name, sim, vehicle, reward);
                sb.Append(i + 1 < catalog.Count ? ",\n" : "\n");
            }
            sb.Append("  ],\n");
            sb.Append("  \"procedural_refs\": [\n");
            for (int i = 0; i < ProceduralRefSeeds.Length; i++)
            {
                ProceduralTrackGenerator.Result r = ProceduralTrackGenerator.Generate(ProceduralRefSeeds[i]);
                try
                {
                    AppendEntry(sb, r.Definition, -1, null, sim, vehicle, reward);
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(r.Definition);
                }
                sb.Append(i + 1 < ProceduralRefSeeds.Length ? ",\n" : "\n");
            }
            sb.Append("  ]\n");
            sb.Append("}\n");
            return sb.ToString();
        }

        static void AppendEntry(StringBuilder sb, TrackDefinition def, int index, string asset, SimConfig sim, VehicleConfig vehicle,
                                RewardConfig reward)
        {
            var c = CultureInfo.InvariantCulture;
            var g = new TrackGeometry(def);
            sb.Append("    {\"index\": ").Append(index.ToString(c));
            sb.Append(", \"id\": \"").Append(def.trackId).Append('"');
            if (asset != null) sb.Append(", \"asset\": \"").Append(asset).Append('"');
            sb.Append(", \"profile\": \"").Append(def.profile.ToString()).Append('"');
            sb.Append(", \"width\": ").Append(def.width.ToString("R", c));
            sb.Append(", \"length_m\": ").Append(g.Length.ToString("R", c));
            sb.Append(", \"checkpoints\": ").Append(g.CheckpointCount.ToString(c));
            sb.Append(", \"half_width\": ").Append(g.HalfWidth.ToString("R", c));
            sb.Append(", \"elevation\": ").Append(def.HasElevation ? "true" : "false");
            sb.Append(", \"env_config_hash\": \"").Append(RaceEnvironment.ComputeEnvConfigHash(sim, vehicle, reward, def)).Append('"');
            sb.Append(", \"track_hash\": \"").Append(ConfigHash.Compute(def)).Append("\"}");
        }

        /// <summary>Writes python/racing_rl/bridge/track_catalog.json (UTF-8 without BOM). Editor-only: the player build is unaffected.</summary>
        [MenuItem("Racing/Tracks/Export Track Catalog (M6)")]
        public static string ExportCatalogJson()
        {
            string json = BuildCatalogJson();
            File.WriteAllText(CatalogJsonPath, json, new UTF8Encoding(false));
            Debug.Log("[Race] track catalog exported: " + CatalogJsonRelPath);
            return json;
        }
    }
}
