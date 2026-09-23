using System.Collections.Generic;
using System.Globalization;
using NUnit.Framework;
using Racing.Core;
using UnityEngine;

namespace Racing.Tests
{
    public class AutopilotLapTests
    {
        RaceEnvironment _env;

        [TearDown]
        public void TearDown() => TestUtil.Destroy(_env);

        [Test]
        public void PurePursuit_Completes3Laps_WithoutWallContact()
        {
            _env = TestUtil.CreateEnv(1, 1000, StartMode.EvalGrid);
            RaceAgentCore agent = _env.Agents[0];
            var pilot = new PurePursuitInputSource(agent);
            int n = _env.Track.CheckpointCount;
            var laps = new List<float>();
            int wallSteps = 0, wrongWay = 0, steps = 0;
            float maxSpeed = 0f;
            while (laps.Count < 3 && steps < 40000)
            {
                agent.ApplyAction(pilot.ReadAction());
                AgentStepResult r = _env.PhysicsStep()[0];
                steps++;
                if (r.Signals.WallContact) wallSteps++;
                if (r.Signals.Cp == CheckpointEvent.WrongWay || r.Signals.WrongWayHeading) wrongWay++;
                Assert.IsFalse(r.Signals.NonFinite || r.Signals.Flipped || r.Signals.OutOfBounds, "failure signal at step " + steps);
                Assert.IsFalse(r.Signals.NoProgressTimeout, "stuck at step " + steps);
                maxSpeed = Mathf.Max(maxSpeed, agent.Telemetry.Speed);
                if (r.LapCompleted) laps.Add(r.LapTime);
            }
            var c = CultureInfo.InvariantCulture;
            TestContext.WriteLine(string.Format(c, "PurePursuit laps: {0} | max speed {1:F1} km/h | steps {2}",
                string.Join(", ", laps.ConvertAll(x => x.ToString("F2", c))), maxSpeed * 3.6f, steps));
            Assert.AreEqual(3, laps.Count, "3 laps");
            Assert.AreEqual(0, wallSteps, "wall contact steps");
            Assert.AreEqual(0, wrongWay, "wrong way events");
            Assert.AreEqual(3 * n, agent.Tracker.GatesPassed, "all gates, in order");
            Assert.Less(Mathf.Abs(laps[2] - laps[1]) / laps[1], 0.005f, "flying laps consistent within 0.5%");
        }
    }

    public class FuzzTests
    {
        RaceEnvironment _env;

        [TearDown]
        public void TearDown() => TestUtil.Destroy(_env);

        [Test]
        public void RandomActions_16Agents_20kSteps_ObservationsFiniteAndInRange()
        {
            const int agents = 16, totalSteps = 20000;
            _env = TestUtil.CreateEnv(agents, 7, StartMode.TrainRandom);
            int k = _env.Sim.decisionPeriod;
            var rng = new DeterministicRng(99);
            var actions = new VehicleAction[agents];
            var obs = new float[ObservationSpec.Size];
            int wallSignals = 0, nonFinite = 0, resets = 0, obsChecks = 0, flips = 0, oob = 0, stuck = 0;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int step = 0; step < totalSteps; step++)
            {
                if (step % k == 0)
                {
                    for (int i = 0; i < agents; i++)
                    {
                        _env.Agents[i].WriteObservation(obs, 0);
                        obsChecks++;
                        for (int j = 0; j < ObservationSpec.Size; j++)
                        {
                            Assert.IsTrue(float.IsFinite(obs[j]), $"obs[{j}] not finite (agent {i}, step {step})");
                            Assert.That(obs[j], Is.InRange(ObservationSpec.Low(j), ObservationSpec.High(j)), $"obs[{j}] out of range (agent {i}, step {step})");
                        }
                        actions[i] = new VehicleAction(rng.Range(-1f, 1f), rng.Range(-0.3f, 1f));
                    }
                }
                for (int i = 0; i < agents; i++) _env.Agents[i].ApplyAction(actions[i]);
                AgentStepResult[] results = _env.PhysicsStep();
                for (int i = 0; i < agents; i++)
                {
                    EpisodeSignals s = results[i].Signals;
                    if (s.WallContact) wallSignals++;
                    if (s.NonFinite) nonFinite++;
                    if (s.Flipped) flips++;
                    if (s.OutOfBounds) oob++;
                    if (s.NoProgressTimeout) stuck++;
                    // emulate M2 termination so agents keep exploring
                    if (s.WallContact || s.Flipped || s.OutOfBounds || s.NonFinite || s.NoProgressTimeout || s.WrongWayHeading || s.Cp == CheckpointEvent.WrongWay)
                    {
                        _env.ResetAgent(i);
                        resets++;
                    }
                }
            }
            sw.Stop();
            TestContext.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "fuzz: {0} steps x {1} agents in {2:F1}s ({3:F0} agent-steps/s) | wallSignals={4} flips={5} oob={6} stuck={7} nonFinite={8} resets={9} obsChecks={10}",
                totalSteps, agents, sw.Elapsed.TotalSeconds, totalSteps * agents / sw.Elapsed.TotalSeconds, wallSignals, flips, oob, stuck, nonFinite, resets, obsChecks));
            Assert.AreEqual(0, nonFinite, "PhysicsError / non-finite signals");
            Assert.Greater(wallSignals, 0, "wall contact detection must fire under random driving");
        }
    }

    public class DeterminismTests
    {
        static List<Vector3> Run(int steps, out List<Quaternion> rotations, out List<float> obsTrace)
        {
            var positions = new List<Vector3>();
            rotations = new List<Quaternion>();
            obsTrace = new List<float>();
            RaceEnvironment env = TestUtil.CreateEnv(4, 42, StartMode.TrainRandom);
            try
            {
                var rng = new DeterministicRng(123);
                var actions = new VehicleAction[4];
                var obs = new float[ObservationSpec.Size];
                for (int step = 0; step < steps; step++)
                {
                    if (step % 5 == 0)
                    {
                        for (int i = 0; i < 4; i++)
                        {
                            env.Agents[i].WriteObservation(obs, 0);
                            obsTrace.AddRange(obs);
                            actions[i] = new VehicleAction(rng.Range(-1f, 1f), rng.Range(-0.5f, 1f));
                        }
                    }
                    for (int i = 0; i < 4; i++) env.Agents[i].ApplyAction(actions[i]);
                    AgentStepResult[] res = env.PhysicsStep();
                    for (int i = 0; i < 4; i++)
                    {
                        positions.Add(env.Agents[i].State.Position);
                        rotations.Add(env.Agents[i].State.Rotation);
                        EpisodeSignals s = res[i].Signals;
                        if (s.WallContact || s.Flipped || s.OutOfBounds || s.NoProgressTimeout) env.ResetAgent(i);
                    }
                }
            }
            finally
            {
                TestUtil.Destroy(env);
            }
            return positions;
        }

        [Test]
        public void SameSeedAndActions_BitwiseIdenticalTrajectories()
        {
            List<Vector3> a = Run(1000, out List<Quaternion> ra, out List<float> oa);
            List<Vector3> b = Run(1000, out List<Quaternion> rb, out List<float> ob);
            Assert.AreEqual(a.Count, b.Count);
            for (int i = 0; i < a.Count; i++)
            {
                Assert.IsTrue(a[i].x == b[i].x && a[i].y == b[i].y && a[i].z == b[i].z, $"position differs at sample {i}: {a[i]} vs {b[i]}");
                Assert.IsTrue(ra[i].x == rb[i].x && ra[i].y == rb[i].y && ra[i].z == rb[i].z && ra[i].w == rb[i].w, $"rotation differs at sample {i}");
            }
            Assert.AreEqual(oa.Count, ob.Count);
            for (int i = 0; i < oa.Count; i++) Assert.IsTrue(oa[i] == ob[i], $"observation differs at {i}");
            TestContext.WriteLine($"compared {a.Count} poses and {oa.Count} observation floats: identical");
        }
    }

    public class AllocationTests
    {
        RaceEnvironment _env;

        [TearDown]
        public void TearDown() => TestUtil.Destroy(_env);

        /// <summary>
        /// GC.GetTotalMemory is process-wide (editor background threads allocate sporadically), so the
        /// test runs several trials and uses the smallest heap growth: our own per-step allocations would
        /// show up in every trial. Mono grows the heap in 8 KB blocks, hence the small tolerance.
        /// </summary>
        [Test]
        public void PhysicsStepAndObservation_DoNotAllocate()
        {
            _env = TestUtil.CreateEnv(16, 5, StartMode.TrainRandom);
            var obs = new float[ObservationSpec.Size];
            var pilots = new PurePursuitInputSource[16];
            for (int i = 0; i < 16; i++) pilots[i] = new PurePursuitInputSource(_env.Agents[i]);
            void Step()
            {
                for (int i = 0; i < 16; i++) _env.Agents[i].ApplyAction(pilots[i].ReadAction());
                _env.PhysicsStep();
                for (int i = 0; i < 16; i++) _env.Agents[i].WriteObservation(obs, 0);
            }
            for (int i = 0; i < 200; i++) Step(); // warm-up (JIT, first-use caches)

            const int trials = 5, stepsPerTrial = 1000;
            long best = long.MaxValue;
            var deltas = new List<long>();
            for (int t = 0; t < trials; t++)
            {
                int gc0 = System.GC.CollectionCount(0);
                long before = System.GC.GetTotalMemory(false);
                for (int i = 0; i < stepsPerTrial; i++) Step();
                long delta = System.GC.GetTotalMemory(false) - before;
                if (System.GC.CollectionCount(0) != gc0) { deltas.Add(-1); continue; } // a GC ran: trial unusable
                deltas.Add(delta);
                best = System.Math.Min(best, delta);
            }
            TestContext.WriteLine($"heap growth per trial ({stepsPerTrial} steps x 16 agents; -1 = GC ran): {string.Join(", ", deltas)}; min = {best} bytes");
            Assert.AreNotEqual(long.MaxValue, best, "every trial was interrupted by a GC (allocation pressure)");
            Assert.LessOrEqual(best, 8 * 1024, "stepping + observation must not allocate");
        }
    }
}
