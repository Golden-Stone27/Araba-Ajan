using NUnit.Framework;
using Racing.Core;
using UnityEditor;
using UnityEngine;

namespace Racing.Tests
{
    /// <summary>
    /// M6 invariants: Track_A (Benchmark) must stay bit-identical through the multi-track refactor (contracts C0.20).
    /// Golden values were recorded from the M5 code (fddea58) before any track code changed.
    /// </summary>
    public class TrackAGoldenTests
    {
        public const string TrackAPath = "Assets/Racing/Config/TrackDefinition_A.asset";
        public const string GoldenGeometry = "f6930a41d26c9e30";
        public const string GoldenPhysical = "4ee36c524bd1cf90";

        static TrackDefinition LoadA()
        {
            var def = AssetDatabase.LoadAssetAtPath<TrackDefinition>(TrackAPath);
            Assert.IsNotNull(def, TrackAPath);
            return def;
        }

        [Test]
        public void TrackA_Geometry_IsBitIdenticalToM5()
        {
            Assert.AreEqual(GoldenGeometry, TrackFingerprint.Geometry(new TrackGeometry(LoadA())));
        }

        [Test]
        public void TrackA_DefaultDefinition_MatchesAsset()
        {
            Assert.AreEqual(GoldenGeometry, TrackFingerprint.Geometry(new TrackGeometry(TrackDefinition.CreateDefault())));
        }

        [Test]
        public void TrackA_BuiltColliders_AreBitIdenticalToM5()
        {
            var go = new GameObject("GoldenTrackA");
            try
            {
                var rt = go.AddComponent<TrackRuntime>();
                rt.Build(LoadA());
                Assert.AreEqual(1144, rt.WallColliderCount);
                Assert.AreEqual(GoldenPhysical, TrackFingerprint.Physical(rt));
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void TrackA_CanonicalHashPart_IsUnchanged()
        {
            // the frozen env_config_hash (C0.14) is asserted in RewardTests; this pins the track part alone
            Assert.AreEqual("Track_A", LoadA().trackId);
            StringAssert.DoesNotContain("track.elevation", ConfigHash.CanonicalJson(LoadA()));
            StringAssert.DoesNotContain("track.roadCollider", ConfigHash.CanonicalJson(LoadA()));
            StringAssert.DoesNotContain("track.wallCollider", ConfigHash.CanonicalJson(LoadA()));
        }

        [Test]
        public void TrackA_PassesUniversalAndBenchmarkProfile()
        {
            TrackDefinition def = LoadA();
            Assert.AreEqual(TrackProfile.Benchmark, def.profile);
            TrackValidator.Report r = TrackValidator.Validate(new TrackGeometry(def));
            TestContext.WriteLine(r.ToString());
            CollectionAssert.IsEmpty(TrackValidator.CheckProfile(def, r));
        }
    }

    public class TrackLayoutTests
    {
        // stadium: 2 × 200 m straights, 2 × 180° left turns of R 60 (L = 400 + 2πR ≈ 777 m)
        static TrackLayout Stadium(double straight2 = 200.0) =>
            new TrackLayout().Straight(200.0).Left(60.0, 180.0).Straight(straight2).Left(60.0, 180.0);

        static TrackDefinition Def(Vector2[] pts)
        {
            var def = TrackDefinition.CreateDefault();
            def.trackId = "Test";
            def.controlPoints = pts;
            return def;
        }

        [Test]
        public void ClosedLayout_BakesToExpectedGeometry()
        {
            TrackLayout.BakeResult b = Stadium().Bake();
            TestContext.WriteLine(b.ToString());
            Assert.Less(b.Residual, 1e-6);
            var g = new TrackGeometry(Def(b.ControlPoints));
            TrackValidator.Report r = TrackValidator.Validate(g);
            TestContext.WriteLine(r.ToString());
            Assert.AreEqual(b.NominalLength, g.Length, 0.01 * b.NominalLength, "length");
            Assert.AreEqual(360f, r.TotalTurnDeg, 3f, "total turn (counter-clockwise)");
            // the spline overshoots curvature where a straight meets an arc: measured R_min ≈ 0.93 R (Track_A: R70 → 64.7,
            // R15 → 14.0). Layout radii are designed with this margin.
            Assert.That(r.MinRadius, Is.InRange(0.88f * 60f, 60.5f), "arc radius");
            Assert.AreEqual(0f, g.SamplePoint(0).x - b.ControlPoints[0].x, 1e-3f, "s = 0 at the first control point");
        }

        [Test]
        public void FlexStraights_CloseAnOpenLayout()
        {
            // rounded rectangle, legs +x, +z, -x, -z: the -x leg is 30 m and the -z leg 10 m longer than their
            // counterparts; the flex pair (+x, +z) must grow by exactly 30 m and 10 m
            var layout = new TrackLayout().Straight(150.0, true).Left(40.0, 90.0).Straight(100.0, true).Left(40.0, 90.0)
                                          .Straight(180.0).Left(40.0, 90.0).Straight(110.0).Left(40.0, 90.0);
            TrackLayout.BakeResult b = layout.Bake();
            TestContext.WriteLine(b.ToString());
            Assert.Less(b.Residual, 1e-6);
            Assert.AreEqual(30.0, b.FlexDelta0, 1e-6);
            Assert.AreEqual(10.0, b.FlexDelta1, 1e-6);
        }

        [Test]
        public void LayoutMustTurn360()
        {
            var layout = new TrackLayout().Straight(100.0).Left(50.0, 170.0).Straight(100.0).Left(50.0, 180.0);
            Assert.Throws<System.InvalidOperationException>(() => layout.Bake());
        }

        [Test]
        public void UnclosedLayoutWithoutFlex_Throws()
        {
            Assert.Throws<System.InvalidOperationException>(() => Stadium(210.0).Bake());
        }

        [Test]
        public void RightTurnLayout_HasNegativeTotalTurn()
        {
            var layout = new TrackLayout().Straight(200.0).Right(60.0, 180.0).Straight(200.0).Right(60.0, 180.0);
            var g = new TrackGeometry(Def(layout.Bake().ControlPoints));
            Assert.AreEqual(-360f, TrackValidator.Validate(g).TotalTurnDeg, 3f);
        }
    }

    public class TrackCatalogTests
    {
        public const string CatalogPath = "Assets/Racing/Config/TrackCatalog.asset";

        static TrackCatalog Load()
        {
            var c = AssetDatabase.LoadAssetAtPath<TrackCatalog>(CatalogPath);
            Assert.IsNotNull(c, CatalogPath + " (menu Racing/Tracks/Setup Track Catalog)");
            return c;
        }

        [Test]
        public void IndexZero_IsTheBenchmarkAsset()
        {
            TrackCatalog c = Load();
            Assert.GreaterOrEqual(c.Count, 1);
            Assert.AreSame(AssetDatabase.LoadAssetAtPath<TrackDefinition>(TrackAGoldenTests.TrackAPath), c.Get(0));
            Assert.AreEqual(TrackCatalog.BenchmarkId, c.Get(0).trackId);
        }

        [Test]
        public void Ids_AreUniqueAsciiAndResolvable()
        {
            TrackCatalog c = Load();
            var seen = new System.Collections.Generic.HashSet<string>(System.StringComparer.Ordinal);
            for (int i = 0; i < c.Count; i++)
            {
                TrackDefinition def = c.Get(i);
                Assert.IsNotNull(def, "entry " + i);
                StringAssert.IsMatch("^[A-Za-z0-9_]+$", def.trackId);
                Assert.IsTrue(seen.Add(def.trackId), "duplicate id " + def.trackId);
                Assert.IsTrue(c.TryResolve(def.trackId, out TrackDefinition byId, out int idx));
                Assert.AreSame(def, byId);
                Assert.AreEqual(i, idx);
                Assert.IsTrue(c.TryResolve(def.name, out _, out int byName));
                Assert.AreEqual(i, byName);
            }
            Assert.IsFalse(c.TryResolve("NoSuchTrack", out _, out int none));
            Assert.AreEqual(-1, none);
            Assert.IsNull(c.Get(c.Count));
            Assert.IsNull(c.Get(-1));
        }

        [Test]
        public void EveryTrack_PassesUniversalRulesAndItsProfile()
        {
            TrackCatalog c = Load();
            for (int i = 0; i < c.Count; i++)
            {
                TrackDefinition def = c.Get(i);
                TrackValidator.Report r = TrackValidator.Validate(new TrackGeometry(def));
                TestContext.WriteLine($"[{i}] {def.trackId} ({def.profile}) {r}");
                CollectionAssert.IsEmpty(TrackValidator.CheckProfile(def, r), def.trackId);
            }
        }
    }
    /// <summary>Frozen values of the catalog tracks and of reference procedural seeds (contracts C0.20).</summary>
    public class TrackFreezeTests
    {
        static T Load<T>(string path) where T : ScriptableObject
        {
            var a = AssetDatabase.LoadAssetAtPath<T>(path);
            Assert.IsNotNull(a, path);
            return a;
        }

        static string EnvHash(TrackDefinition def) => RaceEnvironment.ComputeEnvConfigHash(
            Load<SimConfig>("Assets/Racing/Config/SimConfig.asset"), Load<VehicleConfig>("Assets/Racing/Config/VehicleConfig.asset"),
            Load<RewardConfig>("Assets/Racing/Config/RewardConfig.asset"), def);

        static void AssertFrozen(TrackDefinition def, string env, string geom, string phys)
        {
            var go = new GameObject("Freeze_" + def.trackId);
            try
            {
                var rt = go.AddComponent<TrackRuntime>();
                rt.Build(def);
                Assert.AreEqual(env, EnvHash(def), def.trackId + " env_config_hash");
                Assert.AreEqual(geom, TrackFingerprint.Geometry(rt.Geometry), def.trackId + " geometry");
                Assert.AreEqual(phys, TrackFingerprint.Physical(rt), def.trackId + " colliders");
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [TestCase(0, "Track_A", "90240ee2b1a58b5b", "f6930a41d26c9e30", "4ee36c524bd1cf90")]
        [TestCase(1, "Track_B", "038a104393cbfb72", "1badf7937c4d3373", "e571cd02540992c9")]
        [TestCase(2, "Track_C", "0e099647315ed638", "73a2aed396375311", "783476a6f82becb7")]
        [TestCase(3, "Track_D", "dbce4b7f772fc7b4", "bfa24bc57d5515dd", "9705cfe460a64ac1")]
        public void CatalogTrack_IsFrozen(int index, string id, string env, string geom, string phys)
        {
            TrackDefinition def = Load<TrackCatalog>(TrackCatalogTests.CatalogPath).Get(index);
            Assert.IsNotNull(def);
            Assert.AreEqual(id, def.trackId);
            AssertFrozen(def, env, geom, phys);
        }

        [TestCase(0L, 6, "00724614a3c71752", "7dc3f2ac61cb7e06", "6ff9369609f9d493")]
        [TestCase(1L, 1, "860ce2361684922d", "a3739e86d59576b9", "62891c418c443ff9")]
        [TestCase(7L, 5, "59966c9f08fc1027", "8395cc520802ccb2", "8e8730edf48f8a74")]
        [TestCase(1000L, 2, "7b4353dd8647e3d3", "9c863a016394b310", "2c72d61f4973a9db")]
        public void ProceduralSeed_IsFrozen(long seed, int attempts, string env, string geom, string phys)
        {
            ProceduralTrackGenerator.Result r = ProceduralTrackGenerator.Generate(seed);
            try
            {
                Assert.AreEqual(attempts, r.Attempts, "accept/reject path");
                AssertFrozen(r.Definition, env, geom, phys);
            }
            finally
            {
                Object.DestroyImmediate(r.Definition);
            }
        }

        [Test]
        public void BakedAssets_MatchTheirLayouts()
        {
            (string path, System.Func<TrackLayout> layout)[] baked =
            {
                ("Assets/Racing/Config/TrackDefinition_B.asset", TrackLayouts.TechnicalB),
                ("Assets/Racing/Config/TrackDefinition_C.asset", TrackLayouts.SpeedwayC),
                ("Assets/Racing/Config/TrackDefinition_D.asset", TrackLayouts.HillD),
            };
            foreach (var (path, layout) in baked)
            {
                Vector2[] expected = layout().Bake().ControlPoints;
                Vector2[] actual = Load<TrackDefinition>(path).controlPoints;
                Assert.AreEqual(expected.Length, actual.Length, path);
                for (int i = 0; i < expected.Length; i++) Assert.AreEqual(expected[i], actual[i], path + " point " + i);
            }
        }
    }

    public class ProceduralTrackTests
    {
        [Test]
        public void SameSeed_GivesBitIdenticalTrack()
        {
            foreach (long seed in new long[] { 3, 42, 123456789 })
            {
                ProceduralTrackGenerator.Result a = ProceduralTrackGenerator.Generate(seed), b = ProceduralTrackGenerator.Generate(seed);
                Assert.AreEqual(a.Attempts, b.Attempts);
                Assert.AreEqual(a.Definition.width, b.Definition.width);
                Assert.AreEqual(TrackFingerprint.Geometry(new TrackGeometry(a.Definition)), TrackFingerprint.Geometry(new TrackGeometry(b.Definition)));
                Assert.AreEqual(ConfigHash.Compute(a.Definition), ConfigHash.Compute(b.Definition));
                Object.DestroyImmediate(a.Definition);
                Object.DestroyImmediate(b.Definition);
            }
        }

        [Test]
        public void Seeds_0To99_AllPassAndAreDistinct()
        {
            var hashes = new System.Collections.Generic.HashSet<string>();
            int cw = 0, maxAttempts = 0;
            for (long seed = 0; seed < 100; seed++)
            {
                ProceduralTrackGenerator.Result r = ProceduralTrackGenerator.Generate(seed);
                TrackDefinition def = r.Definition;
                Assert.AreEqual("proc:" + seed, def.trackId);
                Assert.AreEqual(TrackProfile.Procedural, def.profile);
                CollectionAssert.IsEmpty(TrackValidator.CheckProfile(def, r.Report), def.trackId);
                Assert.IsTrue(hashes.Add(ConfigHash.Compute(def)), "duplicate track for " + def.trackId);
                if (r.Report.TotalTurnDeg < 0f) cw++;
                maxAttempts = Mathf.Max(maxAttempts, r.Attempts);
                Object.DestroyImmediate(def);
            }
            TestContext.WriteLine($"clockwise {cw}/100, max attempts {maxAttempts}");
            Assert.That(cw, Is.InRange(25, 75), "both driving directions");
            Assert.Less(maxAttempts, ProceduralTrackGenerator.MaxAttempts);
        }

        [TestCase("proc:0", true, 0L)]
        [TestCase("proc:9223372036854775807", true, long.MaxValue)]
        [TestCase("proc:", false, 0L)]
        [TestCase("proc:-1", false, 0L)]
        [TestCase("proc:+1", false, 0L)]
        [TestCase("proc: 1", false, 0L)]
        [TestCase("proc:1x", false, 0L)]
        [TestCase("proc:9223372036854775808", false, 0L)]
        [TestCase("PROC:1", false, 0L)]
        [TestCase("Track_A", false, 0L)]
        public void TryParseName(string name, bool ok, long seed)
        {
            Assert.AreEqual(ok, ProceduralTrackGenerator.TryParseName(name, out long parsed));
            if (ok) Assert.AreEqual(seed, parsed);
        }

        [Test]
        public void Catalog_ResolvesProceduralNames()
        {
            var catalog = AssetDatabase.LoadAssetAtPath<TrackCatalog>(TrackCatalogTests.CatalogPath);
            Assert.IsTrue(catalog.TryResolve("proc:7", out TrackDefinition def, out int index));
            Assert.AreEqual(-1, index);
            Assert.AreEqual("proc:7", def.trackId);
            Object.DestroyImmediate(def);
        }
    }
    /// <summary>M6 step 3: elevation profile, sloped geometry and the collider rules of Track_D (C0.20).</summary>
    public class ElevationTests
    {
        const string TrackDPath = "Assets/Racing/Config/TrackDefinition_D.asset";

        static TrackDefinition LoadD()
        {
            var d = AssetDatabase.LoadAssetAtPath<TrackDefinition>(TrackDPath);
            Assert.IsNotNull(d, TrackDPath);
            return d;
        }

        static TrackDefinition Profile(params ElevationKey[] keys)
        {
            var def = TrackDefinition.CreateDefault();
            def.elevation = keys;
            return def;
        }

        [Test]
        public void HeightAt_HitsKeys_UsesCosineSpans_AndWraps()
        {
            TrackDefinition def = Profile(new ElevationKey(0.1f, 0f), new ElevationKey(0.5f, 10f), new ElevationKey(0.7f, 4f));
            Assert.AreEqual(0.0, def.HeightAt(0.1), 1e-6);
            Assert.AreEqual(10.0, def.HeightAt(0.5), 1e-6);
            Assert.AreEqual(4.0, def.HeightAt(0.7), 1e-6);
            Assert.AreEqual(5.0, def.HeightAt(0.3), 1e-6, "cosine midpoint");
            Assert.AreEqual(10.0 * 0.5 * (1.0 - System.Math.Cos(System.Math.PI * 0.25)), def.HeightAt(0.2), 1e-6);
            // wrap span 0.7 -> 1.1 (4 m -> 0 m): midpoint at u = 0.9, and u = 0.0 lies at t = 0.75 of that span
            Assert.AreEqual(2.0, def.HeightAt(0.9), 1e-6);
            Assert.AreEqual(def.HeightAt(0.05), def.HeightAt(1.05), 1e-9, "periodic");
            Assert.AreEqual(4.0 * 0.5 * (1.0 + System.Math.Cos(System.Math.PI * 0.75)), def.HeightAt(0.0), 1e-6);
            Assert.AreEqual(0.0, TrackDefinition.CreateDefault().HeightAt(0.4), "flat without keys");
        }

        [Test]
        public void TrackD_SamplesFollowTheProfile_WithinGradeLimits()
        {
            TrackDefinition def = LoadD();
            var g = new TrackGeometry(def);
            for (int k = 0; k < g.SampleCount; k++)
                Assert.AreEqual((float)def.HeightAt(k * (double)g.SampleSpacing / g.Length), g.SamplePoint(k).y, 1e-4f, "sample " + k);
            TrackValidator.Report r = TrackValidator.Validate(g);
            TestContext.WriteLine(r.ToString());
            Assert.AreEqual(0f, r.MinY, 1e-4f);
            Assert.AreEqual(10f, r.MaxY, 1e-3f);
            Assert.That(r.MaxGrade, Is.InRange(0.03f, 0.06f));
            Assert.AreEqual(0f, g.SamplePoint(0).y, "start line is flat");
            CollectionAssert.IsEmpty(TrackValidator.CheckProfile(def, r));
        }

        [Test]
        public void TrackD_SlopedTangents_HorizontalRights_AndHorizontalProjection()
        {
            var g = new TrackGeometry(LoadD());
            float maxTangentY = 0f;
            for (float s = 3f; s < g.Length; s += 37f)
            {
                Vector3 t = g.TangentAt(s), r = g.RightAt(s);
                maxTangentY = Mathf.Max(maxTangentY, Mathf.Abs(t.y));
                Assert.AreEqual(1f, t.magnitude, 1e-4f);
                Assert.AreEqual(0f, r.y, 1e-6f, "right vector is horizontal");
                Assert.AreEqual(0f, Vector3.Dot(t, r), 1e-4f);

                // a car above the road (any height) projects to the same s / lateral: projection is horizontal
                Vector3 p = g.PointAt(s) + r * 2.5f;
                TrackProjection onRoad = g.Project(p, -1), above = g.Project(p + Vector3.up * 3f, -1);
                Assert.AreEqual(s, onRoad.S, 0.05f, "s at " + s);
                Assert.AreEqual(2.5f, onRoad.Lateral, 0.05f, "lateral at " + s);
                Assert.AreEqual(onRoad.S, above.S);
                Assert.AreEqual(onRoad.Lateral, above.Lateral);
            }
            Assert.Greater(maxTangentY, 0.03f, "tangents follow the slope (reward v.t along the road)");
        }

        [Test]
        public void TrackD_CanonicalJson_ListsTheM6Fields()
        {
            string json = ConfigHash.CanonicalJson(LoadD());
            foreach (string key in new[] { "track.elevation", "track.roadCollider", "track.roadColliderMargin",
                                           "track.wallColliderExtraBelow", "track.wallColliderExtraAbove" })
                StringAssert.Contains("\"" + key + "\"", json);
        }

        [Test]
        public void ElevatedTrack_WithoutStripOrWallExtension_FailsTheUniversalRules()
        {
            TrackDefinition d = Object.Instantiate(LoadD());
            try
            {
                d.wallColliderExtraBelow = 0f;
                d.wallColliderExtraAbove = 0f;
                d.roadCollider = RoadColliderMode.GroundBox;
                var fails = TrackValidator.CheckProfile(d, TrackValidator.Validate(new TrackGeometry(d)));
                TestContext.WriteLine(string.Join("\n", fails));
                Assert.AreEqual(3, fails.Count);
            }
            finally
            {
                Object.DestroyImmediate(d);
            }
        }

        [Test]
        public void TrackD_Colliders_RoadStripUnderTheWalls_TallWalls()
        {
            TrackDefinition def = LoadD();
            var go = new GameObject("TrackD");
            try
            {
                var rt = go.AddComponent<TrackRuntime>();
                rt.Build(def);
                MeshCollider strip = go.GetComponentInChildren<MeshCollider>();
                Assert.IsNotNull(strip, "MeshStrip road collider");
                Assert.AreEqual(RacingLayers.Road, strip.gameObject.layer);
                Assert.IsFalse(strip.convex);

                // every road point is supported by the strip, including under the walls, and the hit faces up
                TrackGeometry g = rt.Geometry;
                Physics.SyncTransforms();
                float reach = g.HalfWidth + def.wallThickness + def.roadColliderMargin - 0.5f;
                for (float s = 0.5f; s < g.Length; s += 13f)
                    foreach (float lat in new[] { -reach, 0f, reach })
                    {
                        Vector3 p = g.PointAt(s) + g.RightAt(s) * lat;
                        Assert.IsTrue(strip.Raycast(new Ray(p + Vector3.up * 2f, Vector3.down), out RaycastHit hit, 4f), $"strip at s={s} lat={lat}");
                        Assert.AreEqual(p.y, hit.point.y, 0.05f);
                        Assert.Greater(hit.normal.y, 0.99f, "up-facing");
                    }

                BoxCollider wall = go.transform.Find("WallRight").GetComponentInChildren<BoxCollider>();
                Assert.AreEqual(RacingLayers.Wall, wall.gameObject.layer);
                Assert.AreEqual(def.wallHeight + 2f * TrackLayouts.HillWallExtension, wall.size.y, 1e-5f);
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }
    }
}
