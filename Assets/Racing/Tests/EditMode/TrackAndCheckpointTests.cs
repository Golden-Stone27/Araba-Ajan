using System.Globalization;
using NUnit.Framework;
using Racing.Core;
using UnityEditor;
using UnityEngine;

namespace Racing.Tests
{
    public class TrackValidatorTests
    {
        static TrackDefinition LoadTrack()
        {
            var def = AssetDatabase.LoadAssetAtPath<TrackDefinition>("Assets/Racing/Config/TrackDefinition_A.asset");
            return def != null ? def : TrackDefinition.CreateDefault();
        }

        [Test]
        public void TrackA_PassesAllValidatorCriteria()
        {
            var g = new TrackGeometry(LoadTrack());
            TrackValidator.Report r = TrackValidator.Validate(g);
            TestContext.WriteLine(r.ToString());
            foreach (var c in r.Corners)
                TestContext.WriteLine(string.Format(CultureInfo.InvariantCulture, "corner s={0:F0}-{1:F0} Rmin={2:F1} turn={3:F0}deg sign={4}", c.StartS, c.EndS, c.MinRadius, c.TurnDeg, c.Sign));
            Assert.That(r.Length, Is.InRange(900f, 1400f), "length");
            Assert.That(r.MinRadius, Is.GreaterThanOrEqualTo(12f), "min radius");
            Assert.That(r.MinSeparation, Is.GreaterThan(r.RequiredSeparation), "self proximity");
            Assert.That(r.LongestStraight, Is.GreaterThanOrEqualTo(200f), "straight");
            Assert.IsTrue(r.HasHairpin, "hairpin");
            Assert.IsTrue(r.HasChicane, "chicane");
            Assert.IsTrue(r.HasFastCorner, "fast corner");
            Assert.IsTrue(r.HasLeft && r.HasRight, "left and right turns");
            Assert.That(r.MaxAbsCoordinate, Is.LessThan(500f), "near origin");
            Assert.IsTrue(r.Passed);
        }

        [Test]
        public void Checkpoints_CountAndOrdering()
        {
            var g = new TrackGeometry(LoadTrack());
            Assert.AreEqual(Mathf.FloorToInt(g.Length / 10f), g.CheckpointCount);
            float prev = -1f;
            for (int i = 0; i < g.CheckpointCount; i++)
            {
                Checkpoint c = g.GetCheckpoint(i);
                Assert.AreEqual(i, c.Index);
                Assert.Greater(c.S, prev);
                Assert.AreEqual(1f, c.Forward.magnitude, 1e-4f);
                Assert.AreEqual(0f, Vector3.Dot(c.Forward, c.Right), 1e-4f);
                prev = c.S;
            }
            Assert.AreEqual(0f, g.GetCheckpoint(0).S);
        }

        [Test]
        public void Projection_RecoversArcPositionAndLateralOffset()
        {
            var g = new TrackGeometry(LoadTrack());
            for (float s = 3f; s < g.Length; s += 97f)
            {
                Vector3 p = g.PointAt(s) + g.RightAt(s) * 2.5f;
                TrackProjection full = g.Project(p, -1);
                TrackProjection hinted = g.Project(p, full.SampleIndex + 5);
                Assert.AreEqual(s, full.S, 0.05f, "s at " + s);
                Assert.AreEqual(2.5f, full.Lateral, 0.05f, "lateral (+ = right) at " + s);
                Assert.AreEqual(full.S, hinted.S, 1e-3f);
            }
        }
    }

    public class CheckpointTrackerTests
    {
        TrackGeometry _circle;

        [SetUp]
        public void SetUp()
        {
            // counter-clockwise circle R = 100 m (L ≈ 628 m, 62 gates)
            var def = TrackDefinition.CreateDefault();
            var pts = new Vector2[48];
            for (int i = 0; i < pts.Length; i++)
            {
                float a = i * Mathf.PI * 2f / pts.Length;
                pts[i] = new Vector2(100f * Mathf.Cos(a), 100f * Mathf.Sin(a));
            }
            def.controlPoints = pts;
            _circle = new TrackGeometry(def);
        }

        Vector3 At(float s, float lateral = 0f) => _circle.PointAt(s) + _circle.RightAt(s) * lateral;
        float GateS(int i) => _circle.GetCheckpoint(i).S;

        [Test]
        public void ForwardCrossing_IncrementsOnce()
        {
            var t = new CheckpointTracker(_circle);
            t.Reset(5, false);
            Assert.AreEqual(CheckpointEvent.Passed, t.Update(At(GateS(5) - 1f), At(GateS(5) + 1f)));
            Assert.AreEqual(6, t.NextIndex);
            Assert.AreEqual(1, t.GatesPassed);
            Assert.AreEqual(CheckpointEvent.None, t.Update(At(GateS(5) + 1f), At(GateS(5) + 2f)));
        }

        [Test]
        public void BackwardCrossing_ReportsWrongWay()
        {
            var t = new CheckpointTracker(_circle);
            t.Reset(6, false);
            Assert.AreEqual(CheckpointEvent.WrongWay, t.Update(At(GateS(5) + 1f), At(GateS(5) - 1f)));
            Assert.AreEqual(6, t.NextIndex);
        }

        [Test]
        public void Oscillation_OnSameGate_CountedOnlyOnce()
        {
            var t = new CheckpointTracker(_circle);
            t.Reset(5, false);
            Vector3 before = At(GateS(5) - 0.5f), after = At(GateS(5) + 0.5f);
            Assert.AreEqual(CheckpointEvent.Passed, t.Update(before, after));
            Assert.AreEqual(CheckpointEvent.WrongWay, t.Update(after, before));
            Assert.AreEqual(CheckpointEvent.None, t.Update(before, after));
            Assert.AreEqual(1, t.GatesPassed);
        }

        [Test]
        public void Teleport_SkippedGatesAreNotCounted()
        {
            var t = new CheckpointTracker(_circle);
            t.Reset(5, false);
            t.Update(At(GateS(5) - 1f), At(GateS(30)));
            Assert.LessOrEqual(t.GatesPassed, 1);
            Assert.LessOrEqual(t.NextIndex, 6);
        }

        [Test]
        public void CrossingOutsideGateWidth_IsIgnored()
        {
            var t = new CheckpointTracker(_circle);
            t.Reset(5, false);
            Assert.AreEqual(CheckpointEvent.None, t.Update(At(GateS(5) - 1f, 9f), At(GateS(5) + 1f, 9f)));
            Assert.AreEqual(5, t.NextIndex);
        }

        [Test]
        public void FullLapFromStart_CompletesExactlyOneLap()
        {
            var t = new CheckpointTracker(_circle);
            float s0 = SpawnSampler.EvalStartS;
            t.Reset(_circle.NextGateAfter(s0), true);
            int passed = 0, laps = 0;
            Vector3 prev = At(s0);
            for (float s = s0 + 0.5f; s <= _circle.Length + s0 + 0.01f; s += 0.5f)
            {
                Vector3 cur = At(s);
                CheckpointEvent e = t.Update(prev, cur);
                if (e == CheckpointEvent.Passed) passed++;
                if (e == CheckpointEvent.LapCompleted) laps++;
                Assert.AreNotEqual(CheckpointEvent.WrongWay, e);
                prev = cur;
            }
            Assert.AreEqual(1, laps);
            Assert.AreEqual(_circle.CheckpointCount - 1, passed);
            Assert.AreEqual(1, t.LapsCompleted);
        }

        [Test]
        public void MidTrackSpawn_FirstStartLineCrossingOnlyStartsTiming()
        {
            var t = new CheckpointTracker(_circle);
            float s0 = GateS(20) + 3f;
            t.Reset(_circle.NextGateAfter(s0), false);
            int laps = 0;
            Vector3 prev = At(s0);
            for (float s = s0 + 0.5f; s <= s0 + 2f * _circle.Length; s += 0.5f)
            {
                Vector3 cur = At(s);
                if (t.Update(prev, cur) == CheckpointEvent.LapCompleted) laps++;
                prev = cur;
            }
            Assert.AreEqual(1, laps, "partial first lap must not count");
        }
    }

    public class ContractTests
    {
        [Test]
        public void ObservationLayoutHash_MatchesContract()
        {
            Assert.AreEqual(26, ObservationSpec.Size);
            Assert.AreEqual("b40ca79bdba1c2c2", ObservationSpec.LayoutHash);
            for (int i = 0; i < ObservationSpec.Size; i++) Assert.Less(ObservationSpec.Low(i), ObservationSpec.High(i));
            Assert.AreEqual(0f, RaySensor.AngleDeg(7), 1e-5f, "centre ray index 7 points forward");
            Assert.AreEqual(-90f, RaySensor.AngleDeg(0), 1e-5f);
            Assert.AreEqual(90f, RaySensor.AngleDeg(14), 1e-5f);
        }

        [Test]
        public void DeterministicRng_IsReproducibleAndInRange()
        {
            var a = new DeterministicRng(42);
            var b = new DeterministicRng(42);
            var c = new DeterministicRng(43);
            bool differs = false;
            for (int i = 0; i < 1000; i++)
            {
                float x = a.NextFloat();
                Assert.AreEqual(x, b.NextFloat());
                if (x != c.NextFloat()) differs = true;
                Assert.That(x, Is.GreaterThanOrEqualTo(0f).And.LessThan(1f));
            }
            Assert.IsTrue(differs);
        }

        [Test]
        public void ConfigHash_IsCultureInvariant()
        {
            var sim = SimConfig.CreateDefault();
            var saved = CultureInfo.CurrentCulture;
            try
            {
                CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
                string invariant = ConfigHash.Compute(sim);
                CultureInfo.CurrentCulture = new CultureInfo("tr-TR");
                string turkish = ConfigHash.Compute(sim);
                Assert.AreEqual(invariant, turkish);
                StringAssert.DoesNotContain(",02", ConfigHash.CanonicalJson(sim));
            }
            finally
            {
                CultureInfo.CurrentCulture = saved;
            }
        }

        [Test]
        public void VehicleAction_SanitizeClampsAndZeroesNonFinite()
        {
            var a = new VehicleAction(float.NaN, 3f).Sanitized();
            Assert.AreEqual(0f, a.Steer);
            Assert.AreEqual(1f, a.Throttle);
            var b = new VehicleAction(-7f, float.PositiveInfinity).Sanitized();
            Assert.AreEqual(-1f, b.Steer);
            Assert.AreEqual(0f, b.Throttle);
        }
    }
}
