using UnityEngine;

namespace Racing.Core
{
    /// <summary>
    /// Baked control points (x, z in metres) for Track_A. Generated from a straight/arc layout
    /// (260 m straight, R70 sweeper, R30 chicane, R15 hairpin, R40/R35/R45 corners), closed and
    /// centred on the origin. Validated: L≈1144 m, R_min≈14 m, longest straight≈283 m.
    /// </summary>
    public static class TrackPresets
    {
        public static Vector2[] TrackA => (Vector2[])s_trackA.Clone();

        static readonly Vector2[] s_trackA =
        {
            new Vector2(-152.73f, -115.5f), new Vector2(-146.69f, -115.5f), new Vector2(-140.64f, -115.5f), new Vector2(-134.59f, -115.5f),
            new Vector2(-128.55f, -115.5f), new Vector2(-122.5f, -115.5f), new Vector2(-116.45f, -115.5f), new Vector2(-110.41f, -115.5f),
            new Vector2(-104.36f, -115.5f), new Vector2(-98.31f, -115.5f), new Vector2(-92.27f, -115.5f), new Vector2(-86.22f, -115.5f),
            new Vector2(-80.17f, -115.5f), new Vector2(-74.13f, -115.5f), new Vector2(-68.08f, -115.5f), new Vector2(-62.03f, -115.5f),
            new Vector2(-55.99f, -115.5f), new Vector2(-49.94f, -115.5f), new Vector2(-43.9f, -115.5f), new Vector2(-37.85f, -115.5f),
            new Vector2(-31.8f, -115.5f), new Vector2(-25.76f, -115.5f), new Vector2(-19.71f, -115.5f), new Vector2(-13.66f, -115.5f),
            new Vector2(-7.62f, -115.5f), new Vector2(-1.57f, -115.5f), new Vector2(4.48f, -115.5f), new Vector2(10.52f, -115.5f),
            new Vector2(16.57f, -115.5f), new Vector2(22.62f, -115.5f), new Vector2(28.66f, -115.5f), new Vector2(34.71f, -115.5f),
            new Vector2(40.76f, -115.5f), new Vector2(46.8f, -115.5f), new Vector2(52.85f, -115.5f), new Vector2(58.9f, -115.5f),
            new Vector2(64.94f, -115.5f), new Vector2(70.99f, -115.5f), new Vector2(77.03f, -115.5f), new Vector2(83.08f, -115.5f),
            new Vector2(89.13f, -115.5f), new Vector2(95.17f, -115.5f), new Vector2(101.22f, -115.5f), new Vector2(107.27f, -115.5f),
            new Vector2(113.37f, -115.23f), new Vector2(119.42f, -114.44f), new Vector2(125.38f, -113.11f), new Vector2(131.21f, -111.28f),
            new Vector2(136.85f, -108.94f), new Vector2(142.27f, -106.12f), new Vector2(147.42f, -102.84f), new Vector2(152.26f, -99.12f),
            new Vector2(156.76f, -95.0f), new Vector2(160.89f, -90.5f), new Vector2(164.61f, -85.65f), new Vector2(167.89f, -80.5f),
            new Vector2(170.71f, -75.08f), new Vector2(173.05f, -69.44f), new Vector2(174.88f, -63.62f), new Vector2(176.2f, -57.66f),
            new Vector2(177.0f, -51.6f), new Vector2(177.27f, -45.5f), new Vector2(177.27f, -39.79f), new Vector2(177.27f, -34.07f),
            new Vector2(177.27f, -28.36f), new Vector2(177.27f, -22.64f), new Vector2(177.27f, -16.93f), new Vector2(177.27f, -11.21f),
            new Vector2(177.27f, -5.5f), new Vector2(178.08f, 1.42f), new Vector2(180.46f, 7.96f), new Vector2(184.29f, 13.78f),
            new Vector2(187.5f, 17.61f), new Vector2(190.71f, 21.44f), new Vector2(194.54f, 27.26f), new Vector2(196.92f, 33.81f),
            new Vector2(197.73f, 40.73f), new Vector2(197.73f, 46.73f), new Vector2(197.73f, 52.73f), new Vector2(197.73f, 58.73f),
            new Vector2(197.73f, 64.73f), new Vector2(197.73f, 70.73f), new Vector2(197.73f, 76.73f), new Vector2(197.73f, 82.73f),
            new Vector2(197.73f, 88.73f), new Vector2(197.73f, 94.73f), new Vector2(197.73f, 100.73f), new Vector2(196.83f, 105.86f),
            new Vector2(194.22f, 110.37f), new Vector2(190.23f, 113.72f), new Vector2(185.34f, 115.5f), new Vector2(180.13f, 115.5f),
            new Vector2(175.23f, 113.72f), new Vector2(171.24f, 110.37f), new Vector2(168.64f, 105.86f), new Vector2(167.73f, 100.73f),
            new Vector2(167.73f, 94.48f), new Vector2(167.73f, 88.23f), new Vector2(167.73f, 81.98f), new Vector2(167.73f, 75.73f),
            new Vector2(167.73f, 69.48f), new Vector2(167.73f, 63.23f), new Vector2(167.73f, 56.98f), new Vector2(167.73f, 50.73f),
            new Vector2(167.24f, 44.47f), new Vector2(165.77f, 38.37f), new Vector2(163.37f, 32.57f), new Vector2(160.09f, 27.22f),
            new Vector2(156.02f, 22.44f), new Vector2(151.24f, 18.37f), new Vector2(145.89f, 15.09f), new Vector2(140.09f, 12.69f),
            new Vector2(133.99f, 11.22f), new Vector2(127.73f, 10.73f), new Vector2(121.68f, 10.73f), new Vector2(115.63f, 10.73f),
            new Vector2(109.58f, 10.73f), new Vector2(103.53f, 10.73f), new Vector2(97.48f, 10.73f), new Vector2(91.42f, 10.73f),
            new Vector2(85.37f, 10.73f), new Vector2(79.32f, 10.73f), new Vector2(73.27f, 10.73f), new Vector2(67.22f, 10.73f),
            new Vector2(61.17f, 10.73f), new Vector2(55.12f, 10.73f), new Vector2(49.06f, 10.73f), new Vector2(43.01f, 10.73f),
            new Vector2(36.96f, 10.73f), new Vector2(30.91f, 10.73f), new Vector2(24.86f, 10.73f), new Vector2(18.81f, 10.73f),
            new Vector2(12.76f, 10.73f), new Vector2(6.71f, 10.73f), new Vector2(0.65f, 10.73f), new Vector2(-5.4f, 10.73f),
            new Vector2(-11.45f, 10.73f), new Vector2(-17.5f, 10.73f), new Vector2(-23.55f, 10.73f), new Vector2(-29.6f, 10.73f),
            new Vector2(-35.65f, 10.73f), new Vector2(-41.71f, 10.73f), new Vector2(-47.76f, 10.73f), new Vector2(-53.81f, 10.73f),
            new Vector2(-59.86f, 10.73f), new Vector2(-65.91f, 10.73f), new Vector2(-71.96f, 10.73f), new Vector2(-78.01f, 10.73f),
            new Vector2(-84.06f, 10.73f), new Vector2(-90.12f, 10.73f), new Vector2(-96.17f, 10.73f), new Vector2(-102.22f, 10.73f),
            new Vector2(-108.27f, 10.73f), new Vector2(-114.32f, 10.73f), new Vector2(-120.37f, 10.73f), new Vector2(-126.42f, 10.73f),
            new Vector2(-132.48f, 10.73f), new Vector2(-138.53f, 10.73f), new Vector2(-144.58f, 10.73f), new Vector2(-150.63f, 10.73f),
            new Vector2(-156.68f, 10.73f), new Vector2(-162.73f, 10.73f), new Vector2(-168.81f, 10.2f), new Vector2(-174.7f, 8.62f),
            new Vector2(-180.23f, 6.04f), new Vector2(-185.23f, 2.54f), new Vector2(-189.54f, -1.77f), new Vector2(-193.04f, -6.77f),
            new Vector2(-195.62f, -12.3f), new Vector2(-197.2f, -18.19f), new Vector2(-197.73f, -24.27f), new Vector2(-197.73f, -30.05f),
            new Vector2(-197.73f, -35.83f), new Vector2(-197.73f, -41.61f), new Vector2(-197.73f, -47.39f), new Vector2(-197.73f, -53.16f),
            new Vector2(-197.73f, -58.94f), new Vector2(-197.73f, -64.72f), new Vector2(-197.73f, -70.5f), new Vector2(-197.35f, -76.37f),
            new Vector2(-196.2f, -82.15f), new Vector2(-194.31f, -87.72f), new Vector2(-191.7f, -93.0f), new Vector2(-188.43f, -97.89f),
            new Vector2(-184.55f, -102.32f), new Vector2(-180.13f, -106.2f), new Vector2(-175.23f, -109.47f), new Vector2(-169.95f, -112.07f),
            new Vector2(-164.38f, -113.97f), new Vector2(-158.61f, -115.11f)
        };
    }
}
