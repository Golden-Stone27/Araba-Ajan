using System.Collections.Generic;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>Vehicle dynamics parameters (M1 tunes these; C0.5 action mapping).</summary>
    [CreateAssetMenu(menuName = "Racing/Vehicle Config", fileName = "VehicleConfig")]
    public sealed class VehicleConfig : ScriptableObject, IHashableConfig
    {
        [Header("Body")]
        public float mass = 1200f;
        public Vector3 centerOfMass = new Vector3(0f, 0.35f, 0.1f);
        public float linearDamping = 0.05f;
        public float angularDamping = 0.05f;
        public Vector3 bodySize = new Vector3(1.8f, 0.6f, 4.2f);
        public float bodyCenterY = 0.75f;
        public float halfWidthForClearance = 0.9f;

        [Header("Wheels")]
        public float wheelRadius = 0.34f;
        public float wheelMass = 20f;
        public float wheelX = 0.8f;
        public float wheelY = 0.45f;
        public float wheelZFront = 1.3f;
        public float wheelZRear = -1.3f;
        public float suspensionDistance = 0.2f;
        public float forceAppPointDistance = 0.1f;
        public float spring = 35000f;
        public float damper = 4500f;
        public float targetPosition = 0.5f;

        [Header("Friction (extremumSlip, extremumValue, asymptoteSlip, asymptoteValue, stiffness)")]
        public Vector4 forwardFriction = new Vector4(0.4f, 1f, 0.8f, 0.5f);
        public float forwardStiffness = 1.5f;
        public Vector4 sidewaysFriction = new Vector4(0.2f, 1f, 0.5f, 0.75f);
        public float sidewaysStiffness = 1.5f;

        [Header("Drive / steering (C0.5)")]
        public float maxSteerLowSpeedDeg = 30f;
        public float maxSteerHighSpeedDeg = 8f;
        public float steerSpeedReference = 40f;
        public float steerRateDegPerSec = 200f;
        public float motorTorquePerWheel = 600f;
        public float brakeTorquePerWheel = 3000f;
        public float speedCap = 45f;

        [Header("Aero / stability")]
        public float dragCoefficient = 1.0f;
        public float downforceCoefficient = 2.5f;
        public float antiRollStiffness = 5000f;

        [Header("Spawn")]
        public float spawnHeight = 0.05f;

        public float WheelBase => wheelZFront - wheelZRear;

        public static VehicleConfig CreateDefault() => CreateInstance<VehicleConfig>();

        public void AppendCanonical(SortedDictionary<string, string> kv)
        {
            void V3(string k, Vector3 v) { kv[k + ".x"] = ConfigHash.F(v.x); kv[k + ".y"] = ConfigHash.F(v.y); kv[k + ".z"] = ConfigHash.F(v.z); }
            void V4(string k, Vector4 v) { V3(k, v); kv[k + ".w"] = ConfigHash.F(v.w); }
            kv["veh.mass"] = ConfigHash.F(mass);
            V3("veh.centerOfMass", centerOfMass);
            kv["veh.linearDamping"] = ConfigHash.F(linearDamping);
            kv["veh.angularDamping"] = ConfigHash.F(angularDamping);
            V3("veh.bodySize", bodySize);
            kv["veh.bodyCenterY"] = ConfigHash.F(bodyCenterY);
            kv["veh.halfWidthForClearance"] = ConfigHash.F(halfWidthForClearance);
            kv["veh.wheelRadius"] = ConfigHash.F(wheelRadius);
            kv["veh.wheelMass"] = ConfigHash.F(wheelMass);
            kv["veh.wheelX"] = ConfigHash.F(wheelX);
            kv["veh.wheelY"] = ConfigHash.F(wheelY);
            kv["veh.wheelZFront"] = ConfigHash.F(wheelZFront);
            kv["veh.wheelZRear"] = ConfigHash.F(wheelZRear);
            kv["veh.suspensionDistance"] = ConfigHash.F(suspensionDistance);
            kv["veh.forceAppPointDistance"] = ConfigHash.F(forceAppPointDistance);
            kv["veh.spring"] = ConfigHash.F(spring);
            kv["veh.damper"] = ConfigHash.F(damper);
            kv["veh.targetPosition"] = ConfigHash.F(targetPosition);
            V4("veh.forwardFriction", forwardFriction);
            kv["veh.forwardStiffness"] = ConfigHash.F(forwardStiffness);
            V4("veh.sidewaysFriction", sidewaysFriction);
            kv["veh.sidewaysStiffness"] = ConfigHash.F(sidewaysStiffness);
            kv["veh.maxSteerLowSpeedDeg"] = ConfigHash.F(maxSteerLowSpeedDeg);
            kv["veh.maxSteerHighSpeedDeg"] = ConfigHash.F(maxSteerHighSpeedDeg);
            kv["veh.steerSpeedReference"] = ConfigHash.F(steerSpeedReference);
            kv["veh.steerRateDegPerSec"] = ConfigHash.F(steerRateDegPerSec);
            kv["veh.motorTorquePerWheel"] = ConfigHash.F(motorTorquePerWheel);
            kv["veh.brakeTorquePerWheel"] = ConfigHash.F(brakeTorquePerWheel);
            kv["veh.speedCap"] = ConfigHash.F(speedCap);
            kv["veh.dragCoefficient"] = ConfigHash.F(dragCoefficient);
            kv["veh.downforceCoefficient"] = ConfigHash.F(downforceCoefficient);
            kv["veh.antiRollStiffness"] = ConfigHash.F(antiRollStiffness);
            kv["veh.spawnHeight"] = ConfigHash.F(spawnHeight);
        }
    }
}
