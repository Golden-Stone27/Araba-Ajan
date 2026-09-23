using NUnit.Framework;
using Racing.Core;
using UnityEngine;

namespace Racing.Tests
{
    /// <summary>M2 reward / termination unit tests on synthetic StepContexts (no physics).</summary>
    public class RewardTests
    {
        const int K = 5;
        RewardConfig _cfg;
        SimConfig _sim;
        RewardCalculator _calc;
        TrackGeometry _circle;

        [SetUp]
        public void SetUp()
        {
            _cfg = RewardConfig.CreateDefault();
            _sim = SimConfig.CreateDefault();
            _calc = new RewardCalculator(_cfg, K);
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

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_cfg);
            Object.DestroyImmediate(_sim);
        }

        static StepContext Moving(Vector3 velocity) => new StepContext
        {
            Velocity = velocity, Tangent = Vector3.forward, WallClearance = 5f, Steer = 0f, PrevSteer = 0f
        };

        float Decision(StepContext c)
        {
            float r = 0f;
            for (int j = 0; j < K; j++) r += _calc.Step(c).Total;
            return r;
        }

        [Test]
        public void Default_Weights_MatchContract()
        {
            Assert.AreEqual(0.1f, _cfg.wSpeed);
            Assert.AreEqual(0.05f, _cfg.wCheckpoint);
            Assert.AreEqual(2f, _cfg.wLap);
            Assert.AreEqual(0.05f, _cfg.wWall);
            Assert.AreEqual(0.02f, _cfg.wSmooth);
            Assert.AreEqual(-1f, _cfg.crashPenalty);
            Assert.AreEqual(-0.5f, _cfg.stuckPenalty);
        }

        [TestCase(10f)]
        [TestCase(28f)]
        [TestCase(45f)]
        public void AlongTangent_DecisionReward_Is_wv_v_over_50(float v)
        {
            Assert.AreEqual(_cfg.wSpeed * v / 50f, Decision(Moving(Vector3.forward * v)), 1e-6f);
        }

        [Test]
        public void SpeedTerm_IsClippedAt50mps_AndUsesTangentProjection()
        {
            Assert.AreEqual(_cfg.wSpeed, Decision(Moving(Vector3.forward * 80f)), 1e-6f);
            // lateral (zig-zag) velocity earns nothing
            Assert.AreEqual(0f, Decision(Moving(Vector3.right * 30f)), 1e-6f);
        }

        [Test]
        public void Backwards_IsNegative()
        {
            Assert.Less(Decision(Moving(Vector3.back * 10f)), 0f);
        }

        [Test]
        public void ForwardCar_PositivePerDecision_AwayFromWalls()
        {
            // "suicide invariant": any forward motion away from the walls beats standing still
            Assert.Greater(Decision(Moving(Vector3.forward * 1f)), 0f);
        }

        [Test]
        public void WallProximity_Penalty()
        {
            var c = Moving(Vector3.zero);
            c.WallClearance = 1.5f;
            Assert.AreEqual(0f, Decision(c), 1e-7f);
            c.WallClearance = 0.75f;
            Assert.AreEqual(-0.5f * _cfg.wWall, Decision(c), 1e-6f);
            c.WallClearance = 0f;
            Assert.AreEqual(-_cfg.wWall, Decision(c), 1e-6f);
        }

        [Test]
        public void Smoothness_ChargedOnlyWhenSteerChanges()
        {
            var c = Moving(Vector3.zero);
            c.Steer = 1f;
            c.PrevSteer = -1f;
            Assert.AreEqual(-2f * _cfg.wSmooth, _calc.Step(c).Total, 1e-6f);
            c.PrevSteer = 1f; // action repeat between decisions
            Assert.AreEqual(0f, _calc.Step(c).Total, 1e-7f);
        }

        [Test]
        public void TerminalRewards_FollowC09()
        {
            var c = Moving(Vector3.zero);
            foreach (TermReason r in new[] { TermReason.Wall, TermReason.WrongWay, TermReason.Flip, TermReason.OutOfBounds })
            {
                c.Reason = r;
                Assert.AreEqual(-1f, _calc.Step(c).Terminal, r.ToString());
            }
            c.Reason = TermReason.Stuck;
            Assert.AreEqual(-0.5f, _calc.Step(c).Terminal);
            c.Reason = TermReason.TimeLimit;
            Assert.AreEqual(0f, _calc.Step(c).Terminal);
            c.Reason = TermReason.Finished;
            Assert.AreEqual(0f, _calc.Step(c).Terminal);
        }

        [Test]
        public void PhysicsError_GivesExactlyZero_EvenWithNaNState()
        {
            var c = Moving(new Vector3(float.NaN, 0f, float.PositiveInfinity));
            c.WallClearance = float.NaN;
            c.Reason = TermReason.PhysicsError;
            Assert.AreEqual(0f, _calc.Step(c).Total);
        }

        [Test]
        public void GateOscillation_CheckpointRewardGivenOnce()
        {
            var tracker = new CheckpointTracker(_circle);
            tracker.Reset(5, false);
            float gs = _circle.GetCheckpoint(5).S;
            Vector3 before = _circle.PointAt(gs - 0.5f), after = _circle.PointAt(gs + 0.5f);
            float cpReward = 0f;
            int wrongWay = 0;
            for (int k = 0; k < 10; k++)
            {
                Vector3 a = k % 2 == 0 ? before : after, b = k % 2 == 0 ? after : before;
                CheckpointEvent e = tracker.Update(a, b);
                if (e == CheckpointEvent.WrongWay) wrongWay++;
                var c = Moving(Vector3.zero);
                c.Cp = e;
                cpReward += _calc.Step(c).Checkpoint;
            }
            Assert.AreEqual(_cfg.wCheckpoint, cpReward, 1e-7f);
            Assert.Greater(wrongWay, 0, "backward crossing must report WrongWay (terminal)");
        }

        [Test]
        public void Standing_RewardsZero_ThenStuckAtDecision80()
        {
            var monitor = new EpisodeMonitor(_sim, 0.9f);
            var policy = new TerminationPolicy(_sim);
            float s0 = 50f;
            Vector3 pos = _circle.PointAt(s0);
            var state = new VehicleState
            {
                Position = pos,
                Rotation = Quaternion.LookRotation(_circle.TangentAt(s0), Vector3.up),
                IsFinite = true,
                GroundedWheels = 4
            };
            TrackProjection proj = _circle.Project(pos, -1);

            int stuckDecision = -1;
            float stuckReward = 0f;
            for (int d = 1; d <= 100 && stuckDecision < 0; d++)
            {
                float R = 0f;
                TermReason reason = TermReason.None;
                for (int j = 0; j < K; j++)
                {
                    int step = (d - 1) * K + j + 1;
                    EpisodeSignals sig = monitor.Evaluate(state, proj, _circle, false, CheckpointEvent.None);
                    reason = policy.Evaluate(sig, step, 0, 0);
                    var c = new StepContext { Velocity = Vector3.zero, Tangent = proj.Tangent, WallClearance = sig.WallClearance, Reason = reason };
                    R += _calc.Step(c).Total;
                    if (reason != TermReason.None) break;
                }
                if (reason == TermReason.Stuck)
                {
                    stuckDecision = d;
                    stuckReward = R;
                }
                else
                {
                    Assert.AreEqual(TermReason.None, reason);
                    Assert.AreEqual(0f, R, 1e-6f, "standing still on the centreline earns ~0 (decision " + d + ")");
                }
            }
            Assert.AreEqual(80, stuckDecision);
            Assert.AreEqual(-0.5f, stuckReward, 1e-6f);
        }

        [Test]
        public void TerminationPriority_CrashBeatsTruncation()
        {
            var policy = new TerminationPolicy(_sim);
            int limit = _sim.maxEpisodeDecisions * _sim.decisionPeriod;
            Assert.AreEqual(TermReason.None, policy.Evaluate(default, limit - 1, 0, 0));
            Assert.AreEqual(TermReason.TimeLimit, policy.Evaluate(default, limit, 0, 0));
            Assert.AreEqual(TermReason.Finished, policy.Evaluate(default, 10, 3, 3));
            Assert.AreEqual(TermReason.None, policy.Evaluate(default, 10, 5, 0), "maxLaps 0 = unlimited");
            Assert.AreEqual(TermReason.Wall, policy.Evaluate(new EpisodeSignals { WallContact = true }, limit, 3, 3));
            Assert.AreEqual(TermReason.PhysicsError, policy.Evaluate(new EpisodeSignals { NonFinite = true, WallContact = true }, 1, 0, 0));
            Assert.AreEqual(TermReason.WrongWay, policy.Evaluate(new EpisodeSignals { WrongWayHeading = true }, 1, 0, 0));
            Assert.AreEqual(TermReason.WrongWay, policy.Evaluate(new EpisodeSignals { Cp = CheckpointEvent.WrongWay }, 1, 0, 0));
            Assert.IsTrue(TerminationPolicy.IsTruncation(TermReason.TimeLimit));
            Assert.IsTrue(TerminationPolicy.IsTruncation(TermReason.Finished));
            Assert.IsTrue(TerminationPolicy.IsTermination(TermReason.Stuck));
            Assert.IsFalse(TerminationPolicy.IsTermination(TermReason.None));
        }

        [Test]
        public void RewardConfig_ChangesEnvConfigHash()
        {
            var a = RewardConfig.CreateDefault();
            var b = RewardConfig.CreateDefault();
            b.wSpeed = 0.2f;
            Assert.AreNotEqual(ConfigHash.Compute(a), ConfigHash.Compute(b));
            Object.DestroyImmediate(a);
            Object.DestroyImmediate(b);
        }
    }
}
