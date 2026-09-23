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
}
