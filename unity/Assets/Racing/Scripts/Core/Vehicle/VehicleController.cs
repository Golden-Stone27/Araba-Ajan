using UnityEngine;

namespace Racing.Core
{
    public interface IVehicleController
    {
        /// <summary>Called BEFORE each physics step: steering, torques, aero, anti-roll (C0.5).</summary>
        void ApplyAction(in VehicleAction a, float dt);
        /// <summary>Zeroes velocities and wheel state, then Physics.SyncTransforms().</summary>
        void TeleportTo(in Pose pose, float initialSpeed);
        /// <summary>Read AFTER a physics step.</summary>
        VehicleState ReadState();
        VehicleAction LastApplied { get; }
    }

    /// <summary>WheelCollider car. Wheel order: FL, FR, RL, RR. AWD, no reverse gear.</summary>
    public sealed class VehicleController : MonoBehaviour, IVehicleController
    {
        VehicleConfig _cfg;
        Rigidbody _rb;
        WheelCollider[] _wheels;
        Transform[] _visuals;
        float _steerDeg;
        VehicleAction _last;

        public Rigidbody Body => _rb;
        public VehicleConfig Config => _cfg;
        public VehicleAction LastApplied => _last;
        public float SteerDeg => _steerDeg;
        public WheelCollider[] Wheels => _wheels;

        public void Initialize(VehicleConfig cfg, Rigidbody rb, WheelCollider[] wheels, Transform[] visuals)
        {
            _cfg = cfg;
            _rb = rb;
            _wheels = wheels;
            _visuals = visuals;
        }

        public float MaxSteerDeg(float forwardSpeed) =>
            Mathf.Lerp(_cfg.maxSteerLowSpeedDeg, _cfg.maxSteerHighSpeedDeg, Mathf.Clamp01(forwardSpeed / _cfg.steerSpeedReference));

        public void ApplyAction(in VehicleAction a, float dt)
        {
            VehicleAction act = a.Sanitized();
            _last = act;

            Vector3 v = _rb.linearVelocity;
            Vector3 fwd = _rb.rotation * Vector3.forward;
            float vFwd = Vector3.Dot(v, fwd);

            float target = act.Steer * MaxSteerDeg(vFwd);
            float maxDelta = _cfg.steerRateDegPerSec * dt;
            _steerDeg += Mathf.Clamp(target - _steerDeg, -maxDelta, maxDelta);
            _wheels[0].steerAngle = _steerDeg;
            _wheels[1].steerAngle = _steerDeg;

            float motor = 0f, brake = 0f;
            if (act.Throttle >= 0f)
            {
                motor = act.Throttle * _cfg.motorTorquePerWheel;
                if (vFwd >= _cfg.speedCap) motor = 0f;
            }
            else
            {
                brake = -act.Throttle * _cfg.brakeTorquePerWheel;
            }
            for (int i = 0; i < _wheels.Length; i++)
            {
                _wheels[i].motorTorque = motor;
                _wheels[i].brakeTorque = brake;
            }

            _rb.AddForce(-_cfg.dragCoefficient * v.magnitude * v, ForceMode.Force);
            _rb.AddForce(-(_rb.rotation * Vector3.up) * (_cfg.downforceCoefficient * vFwd * vFwd), ForceMode.Force);

            ApplyAntiRoll(_wheels[0], _wheels[1]);
            ApplyAntiRoll(_wheels[2], _wheels[3]);
        }

        void ApplyAntiRoll(WheelCollider left, WheelCollider right)
        {
            float travelL = 1f, travelR = 1f;
            bool gl = left.GetGroundHit(out WheelHit hl);
            if (gl) travelL = (-left.transform.InverseTransformPoint(hl.point).y - left.radius) / left.suspensionDistance;
            bool gr = right.GetGroundHit(out WheelHit hr);
            if (gr) travelR = (-right.transform.InverseTransformPoint(hr.point).y - right.radius) / right.suspensionDistance;
            float force = (travelL - travelR) * _cfg.antiRollStiffness;
            if (gl) _rb.AddForceAtPosition(left.transform.up * -force, left.transform.position);
            if (gr) _rb.AddForceAtPosition(right.transform.up * force, right.transform.position);
        }

        public void TeleportTo(in Pose pose, float initialSpeed)
        {
            _rb.linearVelocity = Vector3.zero;
            _rb.angularVelocity = Vector3.zero;
            transform.SetPositionAndRotation(pose.position, pose.rotation);
            _rb.position = pose.position;
            _rb.rotation = pose.rotation;
            float wheelSpin = initialSpeed / _cfg.wheelRadius * Mathf.Rad2Deg;
            for (int i = 0; i < _wheels.Length; i++)
            {
                _wheels[i].motorTorque = 0f;
                _wheels[i].brakeTorque = 0f;
                _wheels[i].steerAngle = 0f;
                _wheels[i].rotationSpeed = wheelSpin;
            }
            _steerDeg = 0f;
            _last = VehicleAction.Zero;
            Physics.SyncTransforms();
            if (initialSpeed > 0f) _rb.linearVelocity = pose.rotation * Vector3.forward * initialSpeed;
        }

        public VehicleState ReadState()
        {
            Quaternion rot = _rb.rotation;
            Quaternion inv = Quaternion.Inverse(rot);
            Vector3 pos = _rb.position, vel = _rb.linearVelocity, ang = _rb.angularVelocity;
            int grounded = 0;
            for (int i = 0; i < _wheels.Length; i++) if (_wheels[i].isGrounded) grounded++;
            bool finite = Finite(pos) && Finite(vel) && Finite(ang) &&
                          float.IsFinite(rot.x) && float.IsFinite(rot.y) && float.IsFinite(rot.z) && float.IsFinite(rot.w);
            return new VehicleState
            {
                Position = pos,
                Rotation = rot,
                Velocity = vel,
                LocalVelocity = inv * vel,
                LocalAngularVelocity = inv * ang,
                GroundedWheels = grounded,
                SteerDeg = _steerDeg,
                IsFinite = finite
            };
        }

        static bool Finite(Vector3 v) => float.IsFinite(v.x) && float.IsFinite(v.y) && float.IsFinite(v.z);

        void LateUpdate()
        {
            if (_visuals == null) return;
            for (int i = 0; i < _wheels.Length; i++)
            {
                if (_visuals[i] == null) continue;
                _wheels[i].GetWorldPose(out Vector3 p, out Quaternion q);
                _visuals[i].SetPositionAndRotation(p, q * Quaternion.Euler(0f, 0f, 90f));
            }
        }
    }

    /// <summary>Builds the car GameObject hierarchy from a VehicleConfig (no prefab dependency).</summary>
    public static class VehicleFactory
    {
        static readonly string[] WheelNames = { "Wheel_FL", "Wheel_FR", "Wheel_RL", "Wheel_RR" };

        public static VehicleController Create(VehicleConfig cfg, SimConfig sim, Transform parent, string name, Material bodyMaterial = null)
        {
            var root = new GameObject(name) { layer = RacingLayers.Car };
            root.transform.SetParent(parent, false);

            var rb = root.AddComponent<Rigidbody>();
            rb.mass = cfg.mass;
            rb.linearDamping = cfg.linearDamping;
            rb.angularDamping = cfg.angularDamping;
            rb.interpolation = RigidbodyInterpolation.None;
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

            var body = root.AddComponent<BoxCollider>();
            body.center = new Vector3(0f, cfg.bodyCenterY, 0f);
            body.size = cfg.bodySize;

            var bodyVis = GameObject.CreatePrimitive(PrimitiveType.Cube);
            bodyVis.name = "BodyVisual";
            Object.DestroyImmediate(bodyVis.GetComponent<Collider>());
            bodyVis.transform.SetParent(root.transform, false);
            bodyVis.transform.localPosition = body.center;
            bodyVis.transform.localScale = cfg.bodySize;
            if (bodyMaterial != null) bodyVis.GetComponent<Renderer>().sharedMaterial = bodyMaterial;

            var wheels = new WheelCollider[4];
            var visuals = new Transform[4];
            for (int i = 0; i < 4; i++)
            {
                float x = (i % 2 == 0) ? -cfg.wheelX : cfg.wheelX;
                float z = i < 2 ? cfg.wheelZFront : cfg.wheelZRear;
                var wgo = new GameObject(WheelNames[i]) { layer = RacingLayers.Car };
                wgo.transform.SetParent(root.transform, false);
                wgo.transform.localPosition = new Vector3(x, cfg.wheelY, z);
                var wc = wgo.AddComponent<WheelCollider>();
                wc.radius = cfg.wheelRadius;
                wc.mass = cfg.wheelMass;
                wc.suspensionDistance = cfg.suspensionDistance;
                wc.forceAppPointDistance = cfg.forceAppPointDistance;
                wc.suspensionSpring = new JointSpring { spring = cfg.spring, damper = cfg.damper, targetPosition = cfg.targetPosition };
                wc.forwardFriction = Curve(cfg.forwardFriction, cfg.forwardStiffness);
                wc.sidewaysFriction = Curve(cfg.sidewaysFriction, cfg.sidewaysStiffness);
                wheels[i] = wc;

                var vis = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                vis.name = WheelNames[i] + "_Visual";
                Object.DestroyImmediate(vis.GetComponent<Collider>());
                vis.transform.SetParent(root.transform, false);
                vis.transform.localScale = new Vector3(2f * cfg.wheelRadius, 0.12f, 2f * cfg.wheelRadius);
                visuals[i] = vis.transform;
            }
            wheels[0].ConfigureVehicleSubsteps(sim.substepSpeedThreshold, sim.substepsBelowThreshold, sim.substepsAboveThreshold);
            rb.centerOfMass = cfg.centerOfMass;

            var ctrl = root.AddComponent<VehicleController>();
            ctrl.Initialize(cfg, rb, wheels, visuals);
            root.AddComponent<CollisionSensor>();
            return ctrl;
        }

        static WheelFrictionCurve Curve(Vector4 c, float stiffness) => new WheelFrictionCurve
        {
            extremumSlip = c.x,
            extremumValue = c.y,
            asymptoteSlip = c.z,
            asymptoteValue = c.w,
            stiffness = stiffness
        };
    }
}
