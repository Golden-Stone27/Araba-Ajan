using UnityEngine;

namespace Racing.Core
{
    /// <summary>Continuous action, both components in [-1, 1] (C0.5).</summary>
    public readonly struct VehicleAction
    {
        public readonly float Steer;
        public readonly float Throttle;

        public VehicleAction(float steer, float throttle)
        {
            Steer = steer;
            Throttle = throttle;
        }

        public static readonly VehicleAction Zero = new VehicleAction(0f, 0f);

        public bool IsFinite => float.IsFinite(Steer) && float.IsFinite(Throttle);

        /// <summary>Non-finite components become 0, then both are clamped to [-1, 1].</summary>
        public VehicleAction Sanitized()
        {
            float s = float.IsFinite(Steer) ? Mathf.Clamp(Steer, -1f, 1f) : 0f;
            float t = float.IsFinite(Throttle) ? Mathf.Clamp(Throttle, -1f, 1f) : 0f;
            return new VehicleAction(s, t);
        }
    }

    public struct VehicleState
    {
        public Vector3 Position;
        public Quaternion Rotation;
        public Vector3 Velocity;
        public Vector3 LocalVelocity;
        public Vector3 LocalAngularVelocity;
        public int GroundedWheels;
        public float SteerDeg;
        public bool IsFinite;
    }

    public readonly struct Checkpoint
    {
        public readonly int Index;
        public readonly Vector3 Position;
        public readonly Vector3 Forward;
        public readonly Vector3 Right;
        public readonly float HalfWidth;
        public readonly float S;

        public Checkpoint(int index, Vector3 position, Vector3 forward, Vector3 right, float halfWidth, float s)
        {
            Index = index;
            Position = position;
            Forward = forward;
            Right = right;
            HalfWidth = halfWidth;
            S = s;
        }
    }

    public readonly struct TrackProjection
    {
        public readonly float S;
        public readonly float Lateral;
        public readonly Vector3 Tangent;
        public readonly int SampleIndex;

        public TrackProjection(float s, float lateral, Vector3 tangent, int sampleIndex)
        {
            S = s;
            Lateral = lateral;
            Tangent = tangent;
            SampleIndex = sampleIndex;
        }
    }

    public enum CheckpointEvent : byte { None = 0, Passed = 1, LapCompleted = 2, WrongWay = 3 }

    /// <summary>Termination codes (C0.9). Values are part of the bridge protocol.</summary>
    public enum TermReason : byte
    {
        None = 0, Wall = 1, WrongWay = 2, Stuck = 3, Flip = 4, OutOfBounds = 5,
        TimeLimit = 6, PhysicsError = 7, Finished = 8
    }

    public enum StartMode : uint { TrainRandom = 0, EvalGrid = 1 }

    /// <summary>Raw per-physics-step signals produced by M1. M2 maps them to reward and termination.</summary>
    public struct EpisodeSignals
    {
        public bool WallContact;
        public bool Flipped;
        public bool OutOfBounds;
        public bool NoProgressTimeout;
        public bool WrongWayHeading;
        public bool NonFinite;
        public CheckpointEvent Cp;
        public float WallClearance;
    }

    public struct AgentStepResult
    {
        public float Reward;
        public bool Terminated;
        public bool Truncated;
        public TermReason Reason;
        public bool LapCompleted;
        public float LapTime;
        public EpisodeSignals Signals;
    }

    /// <summary>Per-agent telemetry mirrored by RACE_INFO_V1 (C0 / M3).</summary>
    public struct AgentTelemetry
    {
        public TermReason TermReason;
        public bool LapCompletedThisStep;
        public int Laps;
        public int NextCheckpoint;
        public int EpisodeDecisions;
        public float LastLapTime;
        public float BestLapTime;
        public float Speed;
        public float Progress;
        public float EpisodeReturn;
        public float PosX;
        public float PosZ;
    }

    public readonly struct SpawnSpec
    {
        public readonly float S;
        public readonly float Lateral;
        public readonly float HeadingOffsetDeg;
        public readonly float InitialSpeed;
        public readonly bool StartTiming;

        public SpawnSpec(float s, float lateral, float headingOffsetDeg, float initialSpeed, bool startTiming)
        {
            S = s;
            Lateral = lateral;
            HeadingOffsetDeg = headingOffsetDeg;
            InitialSpeed = initialSpeed;
            StartTiming = startTiming;
        }
    }
}
