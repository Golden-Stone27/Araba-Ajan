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
        [TestCase("Track_D")]
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

        /// <summary>
        /// DoD: on Track_D no ray reads 1.0 while a wall is within 50 m. Reference = the horizontal (XZ) ray against the
        /// wall inner faces (centreline ± W/2). The same lap with M1-height wall colliders (no extension) must miss
        /// walls, which is what the extension fixes.
        /// </summary>
        [Test]
        public void TrackD_Rays_SeeEveryWallWithin50m_OnlyWithTheExtension()
        {
            TrackDefinition hill = Resolve("Track_D");
            int tall = RayMisses(hill, out int checkedTall, out float errTall, out float maxYTall);
            TrackDefinition shortWalls = Object.Instantiate(hill);
            try
            {
                shortWalls.wallColliderExtraBelow = 0f;
                shortWalls.wallColliderExtraAbove = 0f;
                int low = RayMisses(shortWalls, out int checkedLow, out _, out _);
                TestContext.WriteLine($"extended walls: {tall} misses / {checkedTall} rays with a wall < 49 m, max |ray - reference| = {errTall:F3} m, car climbed to y = {maxYTall:F1} m");
                TestContext.WriteLine($"M1-height walls: {low} misses / {checkedLow} rays");
                Assert.Greater(maxYTall, 9f, "the lap covered the hill");
                Assert.Greater(checkedTall, 5000);
                Assert.AreEqual(0, tall, "rays reading 1.0 with a wall within 49 m");
                Assert.Less(errTall, 0.5f, "ray distance agrees with the horizontal reference");
                Assert.Greater(low, 0, "without the extension the horizontal rays pass over/under the walls");
            }
            finally
            {
                Object.DestroyImmediate(shortWalls);
            }
        }

        int RayMisses(TrackDefinition def, out int checkedRays, out float maxAbsErr, out float maxY)
        {
            checkedRays = 0;
            maxAbsErr = 0f;
            maxY = float.MinValue;
            _env = RaceEnvironment.Create(def, TestUtil.Vehicle, TestUtil.Sim, 1, 1000, StartMode.EvalGrid, TestUtil.Reward);
            try
            {
                TrackGeometry g = _env.Track;
                int n = g.SampleCount;
                var faces = new Vector2[2, n];
                for (int k = 0; k < n; k++)
                {
                    Vector3 p = g.SamplePoint(k), r = g.SampleRight(k) * g.HalfWidth;
                    faces[0, k] = new Vector2(p.x - r.x, p.z - r.z);
                    faces[1, k] = new Vector2(p.x + r.x, p.z + r.z);
                }
                RaceAgentCore agent = _env.Agents[0];
                var pilot = new PurePursuitInputSource(agent);
                var obs = new float[ObservationSpec.Size];
                int misses = 0;
                bool lap = false;
                for (int step = 0; step < 20000 && !lap; step++)
                {
                    if (step % 5 == 0)
                    {
                        agent.WriteObservation(obs, 0);
                        Vector3 o3 = agent.Rays.LastOrigin;
                        var o = new Vector2(o3.x, o3.z);
                        maxY = Mathf.Max(maxY, o3.y);
                        for (int i = 0; i < RaySensor.Count; i++)
                        {
                            Vector3 d3 = Quaternion.Euler(0f, agent.Rays.LastYawDeg + RaySensor.AngleDeg(i), 0f) * Vector3.forward;
                            float reference = NearestHit(faces, n, o, new Vector2(d3.x, d3.z));
                            if (reference >= RaySensor.MaxDistance - 1f) continue;
                            checkedRays++;
                            float reading = obs[ObservationSpec.RayStart + i] * RaySensor.MaxDistance;
                            if (reading >= RaySensor.MaxDistance - 1e-3f) misses++;
                            else maxAbsErr = Mathf.Max(maxAbsErr, Mathf.Abs(reading - reference));
                        }
                    }
                    agent.ApplyAction(pilot.ReadAction());
                    AgentStepResult res = _env.PhysicsStep()[0];
                    Assert.IsFalse(res.Terminated, "PurePursuit episode ended: " + res.Reason);
                    lap = res.LapCompleted;
                }
                Assert.IsTrue(lap, "one lap");
                return misses;
            }
            finally
            {
                TestUtil.Destroy(_env);
                _env = null;
            }
        }

        /// <summary>Distance along the 2D ray to the nearest wall inner-face segment (∞ if none).</summary>
        static float NearestHit(Vector2[,] faces, int n, Vector2 o, Vector2 d)
        {
            float best = float.PositiveInfinity;
            for (int side = 0; side < 2; side++)
                for (int k = 0; k < n; k++)
                {
                    Vector2 a = faces[side, k], b = faces[side, (k + 1) % n];
                    if ((a - o).sqrMagnitude > 3600f) continue; // segment beyond 60 m
                    Vector2 e = b - a;
                    float den = d.x * e.y - d.y * e.x;
                    if (Mathf.Abs(den) < 1e-9f) continue;
                    Vector2 ao = a - o;
                    float t = (ao.x * e.y - ao.y * e.x) / den;  // along the ray
                    float u = (ao.x * d.y - ao.y * d.x) / den;  // along the segment
                    if (t > 0f && u >= 0f && u <= 1f && t < best) best = t;
                }
            return best;
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

        /// <summary>
        /// Two environments built one after the other in the same process. Holds for flat tracks only: on sloped ground
        /// (Track_D) PhysX results depend on how many times static track colliders were built before in the process
        /// (period-4 pattern, independent of worker threads; contracts C0.20). The bridge builds its track once per
        /// process, so fresh processes and RESET (TrackD_RebuildAgents_IsBitwiseReproducible) stay deterministic.
        /// </summary>
        [TestCase("Track_B")]
        [TestCase("Track_C")]
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

        /// <summary>Bridge RESET path on the sloped track: same environment, RebuildAgents between runs → identical runs.</summary>
        [UnityEngine.TestTools.UnityTest]
        public System.Collections.IEnumerator TrackD_RebuildAgents_IsBitwiseReproducible()
        {
            _env = CreateEnv("Track_D", 4, 42, StartMode.TrainRandom);
            var runs = new List<List<float>>();
            for (int r = 0; r < 3; r++)
            {
                if (r > 0)
                {
                    _env.RebuildAgents();
                    yield return null; // Destroy completes at the end of the frame, as between bridge requests
                }
                runs.Add(Drive(_env, 2000));
            }
            float maxY = 0f;
            for (int i = 1; i < runs[0].Count; i += ObservationSpec.Size + 7) maxY = Mathf.Max(maxY, runs[0][i]);
            for (int r = 1; r < runs.Count; r++)
            {
                Assert.AreEqual(runs[0].Count, runs[r].Count);
                for (int i = 0; i < runs[0].Count; i++)
                    Assert.IsTrue(runs[0][i] == runs[r][i], $"run {r} differs at value {i}: {runs[0][i]} vs {runs[r][i]}");
            }
            TestContext.WriteLine($"Track_D: 3 runs x {runs[0].Count} floats (poses + observations) identical after RebuildAgents; max car y {maxY:F1} m");
            Assert.Greater(maxY, 3f, "cars reached the hill");
        }

        /// <summary>Random actions (seeded) for every agent; records position, rotation and the observation each decision.</summary>
        static List<float> Drive(RaceEnvironment env, int steps)
        {
            var trace = new List<float>();
            int n = env.Agents.Count;
            var rng = new DeterministicRng(123);
            var actions = new VehicleAction[n];
            var obs = new float[ObservationSpec.Size];
            for (int step = 0; step < steps; step++)
            {
                if (step % 5 == 0)
                    for (int i = 0; i < n; i++)
                    {
                        env.Agents[i].WriteObservation(obs, 0);
                        Vector3 p = env.Agents[i].State.Position;
                        Quaternion q = env.Agents[i].State.Rotation;
                        trace.Add(p.x); trace.Add(p.y); trace.Add(p.z); trace.Add(q.x); trace.Add(q.y); trace.Add(q.z); trace.Add(q.w);
                        trace.AddRange(obs);
                        actions[i] = new VehicleAction(rng.Range(-1f, 1f), rng.Range(-0.5f, 1f));
                    }
                for (int i = 0; i < n; i++) env.Agents[i].ApplyAction(actions[i]);
                AgentStepResult[] res = env.PhysicsStep();
                for (int i = 0; i < n; i++)
                {
                    EpisodeSignals s = res[i].Signals;
                    if (s.WallContact || s.Flipped || s.OutOfBounds || s.NoProgressTimeout) env.ResetAgent(i);
                }
            }
            return trace;
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
