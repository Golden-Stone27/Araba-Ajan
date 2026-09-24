using System.Globalization;
using NUnit.Framework;
using Racing.Core;
using UnityEngine;

namespace Racing.Tests
{
    public class RaySensorTests
    {
        GameObject _ground, _wall, _trigger, _car;

        [TearDown]
        public void TearDown()
        {
            TestUtil.Destroy(_ground); TestUtil.Destroy(_wall); TestUtil.Destroy(_trigger); TestUtil.Destroy(_car);
        }

        [Test]
        public void CenterRay_MeasuresWallAt10m_IgnoresTriggerAndOwnBody()
        {
            SimConfig sim = TestUtil.Sim;
            TestUtil.UseScriptPhysics(sim);
            _ground = TestUtil.CreateGround(200f);
            VehicleController car = VehicleFactory.Create(TestUtil.Vehicle, sim, null, "RayCar");
            _car = car.gameObject;
            car.TeleportTo(new Pose(Vector3.zero, Quaternion.identity), 0f);

            float faceZ = RaySensor.ForwardOffset + 10f;
            _wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _wall.layer = RacingLayers.Wall;
            _wall.transform.position = new Vector3(0f, 1f, faceZ + 0.5f);
            _wall.transform.localScale = new Vector3(4f, 2f, 1f);

            _trigger = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _trigger.layer = RacingLayers.Wall;
            _trigger.GetComponent<Collider>().isTrigger = true;
            _trigger.transform.position = new Vector3(0f, 1f, 7f);
            _trigger.transform.localScale = new Vector3(4f, 2f, 1f);
            Physics.SyncTransforms();

            var rays = new RaySensor();
            var buf = new float[RaySensor.Count];
            rays.Sample(car.ReadState(), buf, 0);
            TestContext.WriteLine("rays: " + string.Join(" ", System.Array.ConvertAll(buf, v => v.ToString("F3", CultureInfo.InvariantCulture))));
            Assert.AreEqual(0.2f, buf[7], 0.002f, "centre ray: 10 m / 50 m");
            Assert.AreEqual(1f, buf[0], 1e-6f, "left ray: no wall");
            Assert.AreEqual(1f, buf[14], 1e-6f, "right ray: no wall");

            Object.DestroyImmediate(_wall);
            Physics.SyncTransforms();
            rays.Sample(car.ReadState(), buf, 0);
            Assert.AreEqual(1f, buf[7], 1e-6f, "no wall => 1.0 (trigger ignored)");
        }
    }

    public class VehicleDynamicsTests
    {
        GameObject _ground, _car;

        [TearDown]
        public void TearDown() { TestUtil.Destroy(_ground); TestUtil.Destroy(_car); }

        [Test]
        public void SpawnSettle_StraightAcceleration_AndBraking()
        {
            SimConfig sim = TestUtil.Sim;
            VehicleConfig cfg = TestUtil.Vehicle;
            float dt = sim.fixedDeltaTime;
            TestUtil.UseScriptPhysics(sim);
            _ground = TestUtil.CreateGround();
            VehicleController car = VehicleFactory.Create(cfg, sim, null, "DynCar");
            _car = car.gameObject;
            car.TeleportTo(new Pose(new Vector3(0f, cfg.spawnHeight, 0f), Quaternion.identity), 0f);

            // 1) settle: 25 steps with zero action
            for (int i = 0; i < 25; i++) { car.ApplyAction(VehicleAction.Zero, dt); Physics.Simulate(dt); }
            VehicleState st = car.ReadState();
            TestContext.WriteLine($"settle: vy={st.Velocity.y:F4} y={st.Position.y:F3} grounded={st.GroundedWheels}");
            Assert.Less(Mathf.Abs(st.Velocity.y), 0.05f, "settled vertical speed");
            Assert.AreEqual(4, st.GroundedWheels);

            // 2) full throttle: monotonic speed, drift, upright, 0-100 km/h time
            float prevV = 0f, t100 = -1f, t = 0f;
            float startX = st.Position.x;
            const float v100 = 100f / 3.6f;
            while (t < 10f)
            {
                car.ApplyAction(new VehicleAction(0f, 1f), dt);
                Physics.Simulate(dt);
                t += dt;
                st = car.ReadState();
                float v = st.LocalVelocity.z;
                if (t <= 5f)
                {
                    Assert.GreaterOrEqual(v, prevV - 0.01f, $"speed not monotonic at t={t:F2}");
                    Assert.Less(Mathf.Abs(st.Position.x - startX), 0.5f, "lateral drift");
                    Assert.Greater(Vector3.Dot(st.Rotation * Vector3.up, Vector3.up), 0.95f, "upright");
                }
                if (t100 < 0f && v >= v100) { t100 = t; break; }
                prevV = v;
            }
            TestContext.WriteLine($"0-100 km/h: {t100:F2}s");
            Assert.That(t100, Is.InRange(4f, 7f), "0-100 km/h time");

            // 3) full brake from 100 km/h
            Vector3 p0 = st.Position;
            int steps = 0;
            while (st.LocalVelocity.z > 0.1f && steps < 1000)
            {
                car.ApplyAction(new VehicleAction(0f, -1f), dt);
                Physics.Simulate(dt);
                st = car.ReadState();
                steps++;
            }
            float dist = Vector3.Distance(p0, st.Position);
            TestContext.WriteLine($"100-0 km/h braking distance: {dist:F1} m ({steps * dt:F2}s)");
            Assert.LessOrEqual(dist, 45f, "braking distance");
        }
    }
}
