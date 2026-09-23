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

        /// <summary>
        /// Track_D "Hill" (W = 12 m, L ≈ 1094 m, counter-clockwise): flat start straight, R60 left, a climbing straight
        /// and right-left esses (R40), R30/120° left on the plateau, a long descending diagonal, R55 left, descending
        /// straight into a braking R35 right and a closing R25/180° left.
        /// </summary>
        public static TrackLayout HillD() => new TrackLayout()
            .Straight(160.0, true).Left(60.0, 90.0).Straight(150.0).Right(40.0, 70.0).Left(40.0, 70.0).Straight(60.0, true)
            .Left(30.0, 120.0).Straight(170.0).Left(55.0, 60.0).Straight(140.0).Right(35.0, 90.0).Left(25.0, 180.0);

        /// <summary>
        /// Track_D height profile as (s in metres along the nominal layout, y in metres); baked to ElevationKey u = s / L.
        /// Flat to 150 m, climb to +10 m by 480 m (≈4.8 %), plateau to 600 m, descend to 0 by 960 m (≈4.4 %), flat to the line.
        /// </summary>
        public static readonly (double s, float y)[] HillDElevation = { (0.0, 0f), (150.0, 0f), (480.0, 10f), (600.0, 10f), (960.0, 0f) };

        /// <summary>Wall collider extension used by Track_D (≥ TrackValidator.RequiredWallExtension(6 %) = 4 m, with margin).</summary>
        public const float HillWallExtension = 10f;
    }
}
