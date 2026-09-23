using System.Collections.Generic;
using System.Globalization;
using NUnit.Framework;
using Racing.Core;
using UnityEngine;

namespace Racing.Tests
{
    /// <summary>M6: every new track is drivable, laps/gates/wrong-way logic holds, and physics stays deterministic (C0.20).</summary>
    public class MultiTrackPlayTests
    {
        RaceEnvironment _env;
        TrackDefinition _proc;

        [TearDown]
        public void TearDown()
        {
            TestUtil.Destroy(_env);
            _env = null;
            if (_proc != null) Object.DestroyImmediate(_proc);
            _proc = null;
        }

        TrackDefinition Resolve(string name)
        {
            TrackCatalog catalog = TestUtil.Load<TrackCatalog>("Assets/Racing/Config/TrackCatalog.asset", () => null);
            Assert.IsNotNull(catalog, "TrackCatalog.asset");
            Assert.IsTrue(catalog.TryResolve(name, out TrackDefinition def, out int index), name);
            if (index < 0) _proc = def;
            return def;
        }

        RaceEnvironment CreateEnv(string track, int agents, long seed, StartMode mode) =>
            RaceEnvironment.Create(Resolve(track), TestUtil.Vehicle, TestUtil.Sim, agents, seed, mode, TestUtil.Reward);

        [TestCase("Track_B")]
        [TestCase("Track_C")]
        [TestCase("proc:0")]
        [TestCase("proc:7")]
        public void PurePursuit_Completes3Laps_WithoutWallContact(string track)
        {
            _env = CreateEnv(track, 1, 1000, StartMode.EvalGrid);
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
            TestContext.WriteLine(string.Format(c, "{0}: L={1:F1} m gates={2} PurePursuit laps: {3} | max speed {4:F1} km/h | steps {5}",
                track, _env.Track.Length, n, string.Join(", ", laps.ConvertAll(x => x.ToString("F2", c))), maxSpeed * 3.6f, steps));
            Assert.AreEqual(3, laps.Count, "3 laps");
            Assert.AreEqual(0, wallSteps, "wall contact steps");
            Assert.AreEqual(0, wrongWay, "wrong way events");
            Assert.AreEqual(3 * n, agent.Tracker.GatesPassed, "all gates, in order");
            Assert.Less(Mathf.Abs(laps[2] - laps[1]) / laps[1], 0.005f, "flying laps consistent within 0.5%");
        }

        /// <summary>Generator-level drivability: PurePursuit laps proc:0..19 cleanly (one lap each, from the grid).</summary>
        [Test]
        public void PurePursuit_LapsProceduralSeeds0To19()
        {
            var failures = new List<string>();
            var c = CultureInfo.InvariantCulture;
            for (long seed = 0; seed < 20; seed++)
            {
                string name = ProceduralTrackGenerator.Name(seed);
                _env = CreateEnv(name, 1, 1000, StartMode.EvalGrid);
                RaceAgentCore agent = _env.Agents[0];
                var pilot = new PurePursuitInputSource(agent);
                float lap = float.NaN;
                string problem = null;
                for (int step = 0; step < 20000 && problem == null && float.IsNaN(lap); step++)
                {
                    agent.ApplyAction(pilot.ReadAction());
                    AgentStepResult r = _env.PhysicsStep()[0];
                    EpisodeSignals s = r.Signals;
                    if (s.WallContact || s.Flipped || s.OutOfBounds || s.NonFinite || s.NoProgressTimeout ||
                        s.WrongWayHeading || s.Cp == CheckpointEvent.WrongWay)
                        problem = "signal at step " + step;
                    if (r.LapCompleted) lap = r.LapTime;
                }
                if (problem == null && float.IsNaN(lap)) problem = "no lap";
                TestContext.WriteLine(string.Format(c, "{0}: L={1:F0} W={2} lap={3:F2} {4}", name, _env.Track.Length, _env.TrackDef.width, lap, problem ?? "ok"));
                if (problem != null) failures.Add(name + ": " + problem);
                TearDown();
            }
            CollectionAssert.IsEmpty(failures);
        }

        [TestCase("Track_C")]
        public void DrivingBackwards_IsWrongWay(string track)
        {
            _env = CreateEnv(track, 1, 1000, StartMode.EvalGrid);
            RaceAgentCore agent = _env.Agents[0];
            // facing backwards on the start straight (no reverse gear): driving "forward" crosses the previous gate backwards
            agent.BeginEpisode(new SpawnSpec(35f, 0f, 180f, 8f, false));
            Physics.SyncTransforms();
            TermReason reason = TermReason.None;
            int steps = 0;
            for (; steps < 1500 && reason == TermReason.None; steps++)
            {
                agent.ApplyAction(new VehicleAction(0f, 0.5f));
                reason = _env.PhysicsStep()[0].Reason;
            }
            TestContext.WriteLine($"{track}: {reason} after {steps} steps");
            Assert.AreEqual(TermReason.WrongWay, reason);
            Assert.Less(steps, 50, "gate crossed backwards within 1 s (not the 2 s heading timer)");
        }

        [TestCase("Track_B")]
        [TestCase("proc:7")]
        public void SameSeedAndActions_BitwiseIdenticalTrajectories(string track)
        {
            List<Vector3> a = Run(track, out List<float> oa), b = Run(track, out List<float> ob);
            Assert.AreEqual(a.Count, b.Count);
            for (int i = 0; i < a.Count; i++)
                Assert.IsTrue(a[i].x == b[i].x && a[i].y == b[i].y && a[i].z == b[i].z, $"position differs at sample {i}: {a[i]} vs {b[i]}");
            for (int i = 0; i < oa.Count; i++)
                Assert.IsTrue(oa[i] == ob[i], $"observation differs at {i}: {oa[i]} vs {ob[i]}");
            TestContext.WriteLine($"{track}: compared {a.Count} poses and {oa.Count} observation floats: identical");
        }

        List<Vector3> Run(string track, out List<float> obsTrace)
        {
            var positions = new List<Vector3>();
            obsTrace = new List<float>();
            RaceEnvironment env = CreateEnv(track, 4, 42, StartMode.TrainRandom);
            try
            {
                var rng = new DeterministicRng(123);
                var actions = new VehicleAction[4];
                var obs = new float[ObservationSpec.Size];
                for (int step = 0; step < 1000; step++)
                {
                    if (step % 5 == 0)
                        for (int i = 0; i < 4; i++)
                        {
                            env.Agents[i].WriteObservation(obs, 0);
                            obsTrace.AddRange(obs);
                            actions[i] = new VehicleAction(rng.Range(-1f, 1f), rng.Range(-0.5f, 1f));
                        }
                    for (int i = 0; i < 4; i++) env.Agents[i].ApplyAction(actions[i]);
                    AgentStepResult[] res = env.PhysicsStep();
                    for (int i = 0; i < 4; i++)
                    {
                        positions.Add(env.Agents[i].State.Position);
                        EpisodeSignals s = res[i].Signals;
                        if (s.WallContact || s.Flipped || s.OutOfBounds || s.NoProgressTimeout) env.ResetAgent(i);
                    }
                }
            }
            finally
            {
                TestUtil.Destroy(env);
                if (_proc != null) Object.DestroyImmediate(_proc);
                _proc = null;
            }
            return positions;
        }
    }
}
