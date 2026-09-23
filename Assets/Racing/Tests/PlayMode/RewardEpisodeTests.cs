using System.Globalization;
using NUnit.Framework;
using Racing.Core;
using UnityEngine;

namespace Racing.Tests
{
    /// <summary>M2: reward + termination wired into RaceAgentCore on the real track.</summary>
    public class RewardEpisodeTests
    {
        RaceEnvironment _env;

        [TearDown]
        public void TearDown() => TestUtil.Destroy(_env);

        [Test]
        public void PurePursuit_OneLap_PositiveReturn_EndsFinished()
        {
            _env = TestUtil.CreateEnv(1, 1000, StartMode.EvalGrid);
            _env.ResetAll(1000, StartMode.EvalGrid, 1);
            RaceAgentCore agent = _env.Agents[0];
            var pilot = new PurePursuitInputSource(agent);
            AgentStepResult r = default;
            float sum = 0f;
            int steps = 0;
            while (steps < 20000)
            {
                agent.ApplyAction(pilot.ReadAction());
                r = _env.PhysicsStep()[0];
                sum += r.Reward;
                steps++;
                if (r.Terminated || r.Truncated) break;
            }
            RewardBreakdown rb = agent.EpisodeRewards;
            TestContext.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "reason={0} steps={1} return={2:F3} speed={3:F3} wall={4:F3} smooth={5:F3} cp={6:F3} lap={7:F3} term={8:F3}",
                r.Reason, steps, sum, rb.Speed, rb.Wall, rb.Smooth, rb.Checkpoint, rb.Lap, rb.Terminal));
            Assert.AreEqual(TermReason.Finished, r.Reason);
            Assert.IsTrue(r.Truncated && !r.Terminated, "Finished is a truncation");
            Assert.IsTrue(r.LapCompleted);
            Assert.Greater(sum, 0f, "PurePursuit lap return");
            Assert.AreEqual(_env.Reward.wLap, rb.Lap, 1e-6f);
            Assert.AreEqual(sum, agent.Telemetry.EpisodeReturn, 1e-3f);
            Assert.AreEqual(TermReason.Finished, agent.Telemetry.TermReason);
        }

        [Test]
        public void ZeroAction_StandingStart_EndsStuckAtDecision80()
        {
            _env = TestUtil.CreateEnv(1, 1000, StartMode.EvalGrid);
            RaceAgentCore agent = _env.Agents[0];
            AgentStepResult r = default;
            int steps = 0;
            float sum = 0f;
            while (steps < 1000)
            {
                agent.ApplyAction(VehicleAction.Zero);
                r = _env.PhysicsStep()[0];
                sum += r.Reward;
                steps++;
                if (r.Terminated || r.Truncated) break;
            }
            TestContext.WriteLine($"reason={r.Reason} steps={steps} return={sum.ToString("F4", CultureInfo.InvariantCulture)}");
            Assert.AreEqual(TermReason.Stuck, r.Reason);
            Assert.AreEqual(400, steps, "8 s = 400 physics steps = decision 80");
            Assert.AreEqual(-0.5f, sum, 0.01f);
        }
    }
}
