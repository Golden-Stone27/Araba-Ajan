namespace Racing.Core
{
    /// <summary>
    /// Maps M1 signals to a single termination reason (C0.9). When several fire on the same step the
    /// priority is PhysicsError &gt; Wall &gt; Flip &gt; OutOfBounds &gt; WrongWay &gt; Stuck &gt; Finished &gt; TimeLimit,
    /// i.e. a crash always wins over a truncation.
    /// </summary>
    public sealed class TerminationPolicy
    {
        readonly int _maxPhysicsSteps;

        public TerminationPolicy(SimConfig sim)
        {
            _maxPhysicsSteps = sim.maxEpisodeDecisions * sim.decisionPeriod;
        }

        /// <param name="physicsSteps">physics steps since BeginEpisode (episodeDecisions = physicsSteps / K)</param>
        /// <param name="maxLaps">eval lap target; 0 = unlimited (training)</param>
        public TermReason Evaluate(in EpisodeSignals s, int physicsSteps, int lapsCompleted, int maxLaps)
        {
            if (s.NonFinite) return TermReason.PhysicsError;
            if (s.WallContact) return TermReason.Wall;
            if (s.Flipped) return TermReason.Flip;
            if (s.OutOfBounds) return TermReason.OutOfBounds;
            if (s.Cp == CheckpointEvent.WrongWay || s.WrongWayHeading) return TermReason.WrongWay;
            if (s.NoProgressTimeout) return TermReason.Stuck;
            if (maxLaps > 0 && lapsCompleted >= maxLaps) return TermReason.Finished;
            if (physicsSteps >= _maxPhysicsSteps) return TermReason.TimeLimit;
            return TermReason.None;
        }

        public static bool IsTruncation(TermReason r) => r == TermReason.TimeLimit || r == TermReason.Finished;
        public static bool IsTermination(TermReason r) => r != TermReason.None && !IsTruncation(r);
    }
}
