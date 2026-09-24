using System;
using System.Globalization;
using NUnit.Framework;
using Racing.Bridge;
using Racing.Core;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Racing.Tests
{
    /// <summary>
    /// M6 bridge: -trackName / -trackIndex parsing and resolution, HELLO track fields and CONFIG expected_track_id
    /// (contracts C0.20). Python side: python/racing_rl/bridge/protocol.py (HELLO_TRACK_FIELDS, ERR_*).
    /// </summary>
    public class BridgeTrackTests
    {
        static TrackCatalog Catalog()
        {
            var c = AssetDatabase.LoadAssetAtPath<TrackCatalog>(TrackCatalogTests.CatalogPath);
            Assert.IsNotNull(c, TrackCatalogTests.CatalogPath);
            return c;
        }

        static TrackDefinition TrackA() => Catalog().Get(0);

        static string[] Args(params string[] flags)
        {
            var all = new string[flags.Length + 1];
            all[0] = "RaceEnv.exe";
            flags.CopyTo(all, 1);
            return all;
        }

        static bool Resolve(string[] args, out TrackDefinition track, out int index, out string error, string editorOverride = null) =>
            BridgeDriver.TryResolveTrack(args, Catalog(), editorOverride, TrackA(), out track, out index, out error);

        // ------------------------------------------------------------------ CLI parsing

        [Test]
        public void ParseTrackArgs_AbsentFlags_GiveNoSelection()
        {
            Assert.IsTrue(CommandLineArgs.TryParseTrackArgs(Args("-bridgePort", "6005", "-numAgents", "16"), out string name, out int index, out string error));
            Assert.IsNull(name);
            Assert.AreEqual(-1, index);
            Assert.IsNull(error);
        }

        [TestCase(new[] { "-trackName", "Track_B" }, "Track_B", -1)]
        [TestCase(new[] { "-trackName", "proc:7" }, "proc:7", -1)]
        [TestCase(new[] { "-trackIndex", "2" }, null, 2)]
        [TestCase(new[] { "-trackIndex", "0", "-trackName", "Track_A" }, "Track_A", 0)]
        [TestCase(new[] { "-numAgents", "4", "-trackName", "Track_D", "-bridgePort", "6005" }, "Track_D", -1)]
        public void ParseTrackArgs_Valid(string[] flags, string expectedName, int expectedIndex)
        {
            Assert.IsTrue(CommandLineArgs.TryParseTrackArgs(Args(flags), out string name, out int index, out string error), error);
            Assert.AreEqual(expectedName, name);
            Assert.AreEqual(expectedIndex, index);
        }

        [TestCase(new[] { "-trackName" }, "-trackName needs a value")]
        [TestCase(new[] { "-trackIndex" }, "-trackIndex needs a non-negative integer, got nothing")]
        [TestCase(new[] { "-trackIndex", "x" }, "got 'x'")]
        [TestCase(new[] { "-trackIndex", "-1" }, "got '-1'")]
        [TestCase(new[] { "-trackIndex", "+1" }, "got '+1'")]
        [TestCase(new[] { "-trackIndex", "1.0" }, "got '1.0'")]
        [TestCase(new[] { "-trackName", "Track_B", "-trackIndex", "one" }, "got 'one'")]
        public void ParseTrackArgs_Invalid(string[] flags, string messagePart)
        {
            Assert.IsFalse(CommandLineArgs.TryParseTrackArgs(Args(flags), out _, out int index, out string error));
            Assert.AreEqual(-1, index);
            StringAssert.Contains(messagePart, error);
        }

        [Test]
        public void ArgOverloads_ReadTheGivenList()
        {
            string[] args = Args("-bridgePort", "6123", "-bridgeTimingLog", "t.bin");
            Assert.AreEqual(6123, CommandLineArgs.GetInt(args, "-bridgePort", 6005));
            Assert.AreEqual(16, CommandLineArgs.GetInt(args, "-numAgents", 16));
            Assert.AreEqual("t.bin", CommandLineArgs.GetString(args, "-bridgeTimingLog"));
            Assert.IsNull(CommandLineArgs.GetString(args, "-trackName"));
        }

        // ------------------------------------------------------------------ resolution

        [Test]
        public void NoFlags_KeepTheSceneTrack_WithItsCatalogIndex()
        {
            Assert.IsTrue(Resolve(Args(), out TrackDefinition track, out int index, out string error), error);
            Assert.AreSame(TrackA(), track, "the serialized track object itself (M5 path unchanged)");
            Assert.AreEqual(0, index);
        }

        [TestCase("Track_A", 0)]
        [TestCase("Track_B", 1)]
        [TestCase("Track_C", 2)]
        [TestCase("Track_D", 3)]
        [TestCase("TrackDefinition_A", 0)]
        [TestCase("TrackDefinition_D", 3)]
        public void TrackName_ResolvesCatalogIdsAndAssetNames(string name, int expectedIndex)
        {
            Assert.IsTrue(Resolve(Args("-trackName", name), out TrackDefinition track, out int index, out string error), error);
            Assert.AreSame(Catalog().Get(expectedIndex), track);
            Assert.AreEqual(expectedIndex, index);
        }

        [TestCase(0)]
        [TestCase(1)]
        [TestCase(2)]
        [TestCase(3)]
        public void TrackIndex_ResolvesCatalogEntries(int i)
        {
            Assert.IsTrue(Resolve(Args("-trackIndex", i.ToString(CultureInfo.InvariantCulture)), out TrackDefinition track, out int index, out string error), error);
            Assert.AreSame(Catalog().Get(i), track);
            Assert.AreEqual(i, index);
        }

        [Test]
        public void BothFlags_NamingTheSameTrack_Resolve()
        {
            Assert.IsTrue(Resolve(Args("-trackName", "Track_C", "-trackIndex", "2"), out TrackDefinition track, out int index, out string error), error);
            Assert.AreSame(Catalog().Get(2), track);
            Assert.AreEqual(2, index);
        }

        [Test]
        public void ProceduralName_GivesIndexMinusOne_AndTheFrozenHash()
        {
            Assert.IsTrue(Resolve(Args("-trackName", "proc:7"), out TrackDefinition track, out int index, out string error), error);
            try
            {
                Assert.AreEqual(-1, index);
                Assert.AreEqual("proc:7", track.trackId);
                Assert.AreEqual("59966c9f08fc1027", RaceEnvironment.ComputeEnvConfigHash(
                    AssetDatabase.LoadAssetAtPath<SimConfig>("Assets/Racing/Config/SimConfig.asset"),
                    AssetDatabase.LoadAssetAtPath<VehicleConfig>("Assets/Racing/Config/VehicleConfig.asset"),
                    AssetDatabase.LoadAssetAtPath<RewardConfig>("Assets/Racing/Config/RewardConfig.asset"), track), "C0.20");
            }
            finally
            {
                Object.DestroyImmediate(track);
            }
        }

        [TestCase(new[] { "-trackName", "Track_Z" }, "unknown track 'Track_Z'")]
        [TestCase(new[] { "-trackName", "track_b" }, "unknown track 'track_b'")] // Ordinal
        [TestCase(new[] { "-trackName", "proc:-1" }, "unknown track 'proc:-1'")]
        [TestCase(new[] { "-trackIndex", "4" }, "-trackIndex 4 is out of range")]
        [TestCase(new[] { "-trackName", "Track_B", "-trackIndex", "2" }, "conflicting -trackName Track_B (index 1) and -trackIndex 2 (Track_C)")]
        [TestCase(new[] { "-trackName", "proc:7", "-trackIndex", "0" }, "conflicting -trackName proc:7 (index -1) and -trackIndex 0 (Track_A)")]
        [TestCase(new[] { "-trackIndex", "x" }, "-trackIndex needs a non-negative integer")]
        public void UnresolvableSelection_IsAnError(string[] flags, string messagePart)
        {
            Assert.IsFalse(Resolve(Args(flags), out TrackDefinition track, out int index, out string error));
            Assert.IsNull(track);
            Assert.AreEqual(-1, index);
            StringAssert.Contains(messagePart, error);
        }

        [Test]
        public void EditorOverride_AppliesOnlyWithoutFlags()
        {
            Assert.IsTrue(Resolve(Args(), out TrackDefinition track, out int index, out _, "Track_C"));
            Assert.AreSame(Catalog().Get(2), track);
            Assert.AreEqual(2, index);
            Assert.IsTrue(Resolve(Args("-trackName", "Track_B"), out track, out index, out _, "Track_C"));
            Assert.AreSame(Catalog().Get(1), track);
            Assert.IsTrue(Resolve(Args("-trackIndex", "3"), out track, out index, out _, "Track_C"));
            Assert.AreSame(Catalog().Get(3), track);
            Assert.IsFalse(Resolve(Args(), out _, out _, out string error, "Track_Z"));
            StringAssert.Contains("unknown track 'Track_Z'", error);
        }

        [Test]
        public void WithoutCatalog_OnlyTheSceneTrackResolves()
        {
            Assert.IsTrue(BridgeDriver.TryResolveTrack(Args(), null, null, TrackA(), out TrackDefinition track, out int index, out _));
            Assert.AreSame(TrackA(), track);
            Assert.AreEqual(-1, index);
            Assert.IsFalse(BridgeDriver.TryResolveTrack(Args("-trackName", "Track_A"), null, null, TrackA(), out _, out _, out string error));
            StringAssert.Contains("no TrackCatalog", error);
        }

        // ------------------------------------------------------------------ HELLO

        [Serializable]
        struct HelloMsg
        {
            public int protocol;
            public string env, unity, build_id;
            public int num_agents, obs_dim, act_dim;
            public string obs_layout_hash, env_config_hash;
            public float fixed_dt;
            public int decision_period, max_episode_decisions;
            public string info_struct, track_id;
            public int track_index;
            public float track_length_m;
            public int track_checkpoints;
            public float track_half_width;
            public string track_hash;
        }

        /// <summary>The M3 HELLO format (BridgeDriver before M6): every old field must stay byte-identical.</summary>
        static string M3Hello(string env, string unity, string build, int n, string envHash, float dt, int k, int maxDec) =>
            string.Format(CultureInfo.InvariantCulture,
                "{{\"protocol\":{0},\"env\":\"{1}\",\"unity\":\"{2}\",\"build_id\":\"{3}\",\"num_agents\":{4},\"obs_dim\":{5},\"act_dim\":{6}," +
                "\"obs_layout_hash\":\"{7}\",\"env_config_hash\":\"{8}\",\"fixed_dt\":{9},\"decision_period\":{10},\"max_episode_decisions\":{11}," +
                "\"info_struct\":\"{12}\"}}",
                1, env, unity, build, n, 26, 2, "b40ca79bdba1c2c2", envHash, dt.ToString("R", CultureInfo.InvariantCulture), k, maxDec, "RACE_INFO_V1");

        [Test]
        public void Hello_KeepsTheM3Fields_AndAppendsTheTrackFields()
        {
            TrackDefinition a = TrackA();
            var g = new TrackGeometry(a);
            string hello = BridgeProtocol.HelloJson(a.name, "6000.4.6f1", "abc", 16, "90240ee2b1a58b5b", 0.02f, 5, 3000,
                BridgeProtocol.TrackInfo.From(a, 0, g));
            string m3 = M3Hello("TrackDefinition_A", "6000.4.6f1", "abc", 16, "90240ee2b1a58b5b", 0.02f, 5, 3000);
            StringAssert.StartsWith(m3.Substring(0, m3.Length - 1) + ",\"track_id\":\"Track_A\",\"track_index\":0,", hello);
            Assert.AreEqual(1, BridgeProtocol.Version, "fields were only appended: PROTOCOL/Version stay 1");

            var h = JsonUtility.FromJson<HelloMsg>(hello);
            Assert.AreEqual("Track_A", h.track_id);
            Assert.AreEqual(0, h.track_index);
            Assert.AreEqual(g.Length, h.track_length_m);
            Assert.AreEqual(114, h.track_checkpoints);
            Assert.AreEqual(6f, h.track_half_width);
            Assert.AreEqual(ConfigHash.Compute(a), h.track_hash);
            Assert.AreEqual("90240ee2b1a58b5b", h.env_config_hash);
            Assert.AreEqual("TrackDefinition_A", h.env);
        }

        [TestCase(0, "Track_A", 1144.1f, 114, 6f, "0f2fbf3481a1a24f")]
        [TestCase(1, "Track_B", 1114.8f, 111, 5.5f, "6ce4f5d0b0d07eda")]
        [TestCase(2, "Track_C", 1262.5f, 126, 6.5f, "ff0ea58517167d6f")]
        [TestCase(3, "Track_D", 1094.4f, 109, 6f, "04f6036ec025016a")]
        public void Hello_TrackFields_MatchTheCatalogTrack(int i, string id, float length, int gates, float halfWidth, string trackHash)
        {
            TrackDefinition def = Catalog().Get(i);
            BridgeProtocol.TrackInfo info = BridgeProtocol.TrackInfo.From(def, i, new TrackGeometry(def));
            Assert.AreEqual(id, info.Id);
            Assert.AreEqual(i, info.Index);
            Assert.AreEqual(length, info.LengthM, 0.06f); // C0.20 table rounds L to 0.1 m
            Assert.AreEqual(gates, info.Checkpoints);
            Assert.AreEqual(halfWidth, info.HalfWidth);
            Assert.AreEqual(ConfigHash.Compute(def), info.Hash);
            Assert.AreEqual(trackHash, info.Hash, "frozen track part (C0.20)");
            var h = JsonUtility.FromJson<HelloMsg>(BridgeProtocol.HelloJson(def.name, "u", "b", 1, "e", 0.02f, 5, 3000, info));
            Assert.AreEqual(id, h.track_id);
            Assert.AreEqual(i, h.track_index);
            Assert.AreEqual(info.LengthM, h.track_length_m, "float R round-trip");
            Assert.AreEqual(gates, h.track_checkpoints);
            Assert.AreEqual(info.Hash, h.track_hash);
        }

        [Test]
        public void Hello_ProceduralTrack_HasIndexMinusOne()
        {
            ProceduralTrackGenerator.Result r = ProceduralTrackGenerator.Generate(7);
            try
            {
                var g = new TrackGeometry(r.Definition);
                var h = JsonUtility.FromJson<HelloMsg>(BridgeProtocol.HelloJson(r.Definition.name, "u", "b", 1, "e", 0.02f, 5, 3000,
                    BridgeProtocol.TrackInfo.From(r.Definition, -1, g)));
                Assert.AreEqual("proc:7", h.env);
                Assert.AreEqual("proc:7", h.track_id);
                Assert.AreEqual(-1, h.track_index);
                Assert.AreEqual(g.CheckpointCount, h.track_checkpoints);
            }
            finally
            {
                Object.DestroyImmediate(r.Definition);
            }
        }

        // ------------------------------------------------------------------ CONFIG

        [Test]
        public void Config_WithoutTrackField_ParsesAsBefore()
        {
            var cfg = JsonUtility.FromJson<BridgeProtocol.ConfigMsg>(
                "{\"expected_obs_layout_hash\": \"b40ca79bdba1c2c2\", \"expected_env_config_hash\": \"90240ee2b1a58b5b\", \"strict\": true}");
            Assert.IsTrue(string.IsNullOrEmpty(cfg.expected_track_id));
            Assert.IsTrue(cfg.strict);
            Assert.IsTrue(BridgeProtocol.CheckConfig(cfg, "b40ca79bdba1c2c2", "90240ee2b1a58b5b", "Track_A", out string code, out _));
            Assert.IsNull(code);
        }

        [Test]
        public void Config_EmptyExpectations_AlwaysPass()
        {
            var cfg = JsonUtility.FromJson<BridgeProtocol.ConfigMsg>("{}");
            Assert.IsFalse(cfg.strict);
            Assert.IsTrue(BridgeProtocol.CheckConfig(cfg, "b40ca79bdba1c2c2", "dbce4b7f772fc7b4", "Track_D", out _, out _));
        }

        [Test]
        public void Config_ExpectedTrackId_MatchingPasses_OtherIsTrackMismatch()
        {
            var cfg = JsonUtility.FromJson<BridgeProtocol.ConfigMsg>("{\"expected_track_id\":\"Track_D\",\"strict\":true}");
            Assert.AreEqual("Track_D", cfg.expected_track_id);
            Assert.IsTrue(BridgeProtocol.CheckConfig(cfg, "b40ca79bdba1c2c2", "dbce4b7f772fc7b4", "Track_D", out _, out _));

            Assert.IsFalse(BridgeProtocol.CheckConfig(cfg, "b40ca79bdba1c2c2", "90240ee2b1a58b5b", "Track_A", out string code, out string msg));
            Assert.AreEqual(BridgeProtocol.ErrTrackMismatch, code);
            Assert.AreEqual("TRACK_MISMATCH", code);
            StringAssert.Contains("track_id Track_A (expected Track_D)", msg);
        }

        [Test]
        public void Config_TrackMismatch_IsReportedBeforeTheHash()
        {
            var cfg = new BridgeProtocol.ConfigMsg
            {
                expected_obs_layout_hash = "b40ca79bdba1c2c2", expected_env_config_hash = "dbce4b7f772fc7b4", expected_track_id = "Track_D", strict = true
            };
            Assert.IsFalse(BridgeProtocol.CheckConfig(cfg, "b40ca79bdba1c2c2", "90240ee2b1a58b5b", "Track_A", out string code, out _));
            Assert.AreEqual(BridgeProtocol.ErrTrackMismatch, code);

            cfg.expected_track_id = "Track_A";
            Assert.IsFalse(BridgeProtocol.CheckConfig(cfg, "b40ca79bdba1c2c2", "90240ee2b1a58b5b", "Track_A", out code, out string msg));
            Assert.AreEqual(BridgeProtocol.ErrHashMismatch, code);
            StringAssert.Contains("env_config_hash 90240ee2b1a58b5b (expected dbce4b7f772fc7b4)", msg);
        }

        [Test]
        public void ErrorCodes_MatchThePythonSide()
        {
            Assert.AreEqual("UNKNOWN_TRACK", BridgeProtocol.ErrUnknownTrack);
            Assert.AreEqual("TRACK_MISMATCH", BridgeProtocol.ErrTrackMismatch);
            Assert.AreEqual("HASH_MISMATCH", BridgeProtocol.ErrHashMismatch);
        }
    }
}
