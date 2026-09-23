namespace Racing.Core
{
    /// <summary>
    /// Source layouts of the M6 catalog tracks (contracts C0.20). The baked control points live in the
    /// TrackDefinition assets (menu Racing/Tracks/Bake Track Assets); an EditMode test keeps them in sync.
    /// Radii are chosen with the spline's ≈0.93 R_min margin (hairpin R15 → ≈14 m ≥ 12 m).
    /// </summary>
    public static class TrackLayouts
    {
        /// <summary>
        /// Track_B "Technical" (W = 11 m, L ≈ 1115 m, counter-clockwise): main straight into a right-left chicane,
        /// tightening left (R45 → R22), right-left-right esses (R45), an infield right hairpin (R15) between two
        /// R25 lefts, a double-apex left (2 × R20), a left-right chicane (R28) and a final R35 left.
        /// </summary>
        public static TrackLayout TechnicalB() => new TrackLayout()
            .Straight(130.0, true).Right(30.0, 35.0).Straight(5.0).Left(30.0, 35.0).Straight(60.0).Left(45.0, 45.0).Left(22.0, 45.0)
            .Straight(40.0, true).Right(45.0, 40.0).Straight(10.0).Left(45.0, 80.0).Straight(10.0).Right(45.0, 40.0).Straight(40.0).Left(30.0, 90.0)
            .Straight(60.0).Left(25.0, 90.0).Straight(40.0).Right(15.0, 180.0).Straight(40.0).Left(25.0, 90.0)
            .Straight(60.0).Left(20.0, 45.0).Straight(15.0).Left(20.0, 45.0)
            .Straight(75.0).Left(28.0, 35.0).Straight(5.0).Right(28.0, 35.0).Straight(75.0).Left(35.0, 90.0);

        /// <summary>
        /// Track_C "Speedway" (W = 13 m, L ≈ 1263 m, counter-clockwise tri-oval): front stretch 115 m + R300/30° dogleg
        /// + 115 m, two R95/165° turns, back straight ≈ 328 m.
        /// </summary>
        public static TrackLayout SpeedwayC() => new TrackLayout()
            .Straight(115.0, true).Left(300.0, 30.0).Straight(115.0).Left(95.0, 165.0).Straight(270.0, true).Left(95.0, 165.0);
    }
}
