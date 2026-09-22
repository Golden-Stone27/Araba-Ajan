using UnityEngine;

namespace Racing.Core
{
    /// <summary>
    /// Geometric gate-crossing tracker (no triggers). Only the expected gate counts; crossing the
    /// previous gate backwards reports WrongWay. A lap is counted when gate 0 is crossed after all
    /// gates of the lap were passed in order.
    /// </summary>
    public sealed class CheckpointTracker
    {
        public const float LateralMargin = 1f;

        readonly ITrack _track;
        int _gatesThisLap;

        public int NextIndex { get; private set; }
        public int LapsCompleted { get; private set; }
        public int GatesPassed { get; private set; }
        public int LastPassedIndex { get; private set; } = -1;
        public bool TimingActive { get; private set; }

        public CheckpointTracker(ITrack track) { _track = track; }

        public void Reset(int nextIndex, bool startTiming)
        {
            NextIndex = nextIndex;
            LapsCompleted = 0;
            GatesPassed = 0;
            LastPassedIndex = -1;
            TimingActive = startTiming;
            _gatesThisLap = startTiming ? 1 : 0;
        }

        public CheckpointEvent Update(Vector3 prev, Vector3 curr)
        {
            int n = _track.CheckpointCount;
            Checkpoint gate = _track.GetCheckpoint(NextIndex);
            if (Crosses(gate, prev, curr, true))
            {
                GatesPassed++;
                LastPassedIndex = NextIndex;
                if (NextIndex == 0)
                {
                    bool lap = TimingActive && _gatesThisLap == n;
                    TimingActive = true;
                    _gatesThisLap = 1;
                    NextIndex = 1 % n;
                    if (lap)
                    {
                        LapsCompleted++;
                        return CheckpointEvent.LapCompleted;
                    }
                    return CheckpointEvent.Passed;
                }
                _gatesThisLap++;
                NextIndex = (NextIndex + 1) % n;
                return CheckpointEvent.Passed;
            }

            Checkpoint previous = _track.GetCheckpoint((NextIndex - 1 + n) % n);
            return Crosses(previous, prev, curr, false) ? CheckpointEvent.WrongWay : CheckpointEvent.None;
        }

        static bool Crosses(in Checkpoint g, Vector3 a, Vector3 b, bool forward)
        {
            float da = Vector3.Dot(a - g.Position, g.Forward);
            float db = Vector3.Dot(b - g.Position, g.Forward);
            bool crossed = forward ? (da < 0f && db >= 0f) : (da >= 0f && db < 0f);
            if (!crossed) return false;
            float t = da / (da - db);
            Vector3 x = Vector3.LerpUnclamped(a, b, t);
            return Mathf.Abs(Vector3.Dot(x - g.Position, g.Right)) <= g.HalfWidth + LateralMargin;
        }
    }

    /// <summary>Lap timer in simulated time (integer step count × dt). Never uses Time.time.</summary>
    public sealed class LapTimer
    {
        readonly float _dt;
        int _steps;

        public bool Running { get; private set; }
        public float Current => _steps * _dt;
        public float Last { get; private set; } = float.NaN;
        public float Best { get; private set; } = float.NaN;

        public LapTimer(float dt) { _dt = dt; }

        public void Reset()
        {
            _steps = 0;
            Running = false;
            Last = float.NaN;
            Best = float.NaN;
        }

        public void Start()
        {
            _steps = 0;
            Running = true;
        }

        public void Tick() { if (Running) _steps++; }

        public float MarkLap()
        {
            float t = Current;
            Last = t;
            if (float.IsNaN(Best) || t < Best) Best = t;
            _steps = 0;
            return t;
        }
    }

    /// <summary>Produces EpisodeSignals every physics step. Decides nothing (M2 maps signals to reward/termination).</summary>
    public sealed class EpisodeMonitor
    {
        readonly SimConfig _sim;
        readonly float _halfCarWidth;
        float _flipTime, _wrongWayTime, _sinceProgress;

        public EpisodeMonitor(SimConfig sim, float halfCarWidth)
        {
            _sim = sim;
            _halfCarWidth = halfCarWidth;
        }

        public void Reset()
        {
            _flipTime = 0f;
            _wrongWayTime = 0f;
            _sinceProgress = 0f;
        }

        public EpisodeSignals Evaluate(in VehicleState s, in TrackProjection p, ITrack track, bool wallContact, CheckpointEvent cp)
        {
            float dt = _sim.fixedDeltaTime;
            var sig = new EpisodeSignals { WallContact = wallContact, Cp = cp };
            sig.NonFinite = !s.IsFinite || s.Position.sqrMagnitude > 1e8f;

            float upDot = Vector3.Dot(s.Rotation * Vector3.up, Vector3.up);
            _flipTime = upDot < 0.3f ? _flipTime + dt : 0f;
            sig.Flipped = _flipTime >= _sim.flipTimeout;

            sig.OutOfBounds = s.Position.y < -5f || Mathf.Abs(p.Lateral) > track.HalfWidth + 3f;

            float cosPsi = Mathf.Cos(ObservationBuilder.SignedHeadingErrorRad(s.Rotation, p.Tangent));
            _wrongWayTime = cosPsi < -0.5f ? _wrongWayTime + dt : 0f;
            sig.WrongWayHeading = _wrongWayTime >= _sim.wrongWayTimeout;

            _sinceProgress = (cp == CheckpointEvent.Passed || cp == CheckpointEvent.LapCompleted) ? 0f : _sinceProgress + dt;
            sig.NoProgressTimeout = _sinceProgress >= _sim.noProgressTimeout;

            sig.WallClearance = track.HalfWidth - Mathf.Abs(p.Lateral) - _halfCarWidth;
            return sig;
        }
    }

    /// <summary>Deterministic spawn sampling (C0.10 eval grid, M2 training randomisation).</summary>
    public static class SpawnSampler
    {
        public const float EvalStartS = 1f;

        public static SpawnSpec Sample(StartMode mode, DeterministicRng rng, ITrack track)
        {
            if (mode == StartMode.EvalGrid)
            {
                float lateral = rng.Range(-0.5f, 0.5f);
                float heading = rng.Range(-2f, 2f);
                return new SpawnSpec(EvalStartS, lateral, heading, 0f, true);
            }
            int n = track.CheckpointCount;
            int gate = (int)(rng.NextUInt() % (uint)n);
            float spacing = track.Length / n;
            float s = track.GetCheckpoint(gate).S + 1f + rng.NextFloat() * (spacing - 2f);
            float lat = rng.Range(-1.8f, 1.8f);
            float hdg = rng.Range(-10f, 10f);
            float v0 = rng.Range(0f, 10f);
            return new SpawnSpec(s, lat, hdg, v0, false);
        }
    }
}
