using System.Globalization;
using System.Text;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>Debug overlay for agent 0: speed, laps, checkpoint, signals and the full 26-float observation.</summary>
    public sealed class RaceHud : MonoBehaviour
    {
        [SerializeField] SimulationDriver driver;
        [SerializeField] bool visible = true;

        readonly float[] _obs = new float[ObservationSpec.Size];
        readonly StringBuilder _sb = new StringBuilder(2048);
        GUIStyle _style;

        static readonly string[] ObsNames =
        {
            "ray0", "ray1", "ray2", "ray3", "ray4", "ray5", "ray6", "ray7", "ray8", "ray9", "ray10", "ray11", "ray12", "ray13", "ray14",
            "v_fwd", "v_lat", "yaw_rate", "sin_psi1", "cos_psi1", "sin_psi2", "cos_psi2", "e_lat", "prev_steer", "prev_thr", "grounded"
        };

        void Update()
        {
            if (Input.GetKeyDown(KeyCode.H)) visible = !visible;
        }

        void OnGUI()
        {
            if (!visible) return;
            if (driver == null) driver = FindAnyObjectByType<SimulationDriver>();
            if (driver == null || driver.Environment == null || !driver.Environment.IsInitialized) return;
            RaceAgentCore a = driver.Environment.Agents[0];
            if (_style == null) _style = new GUIStyle(GUI.skin.box) { alignment = TextAnchor.UpperLeft, fontSize = 13, richText = true };

            a.WriteObservation(_obs, 0);
            var c = CultureInfo.InvariantCulture;
            AgentTelemetry t = a.Telemetry;
            EpisodeSignals s = a.LastSignals;
            _sb.Clear();
            _sb.AppendFormat(c, "<b>{0}</b>  [P] autopilot  [R] reset  [H] HUD\n", driver.Autopilot ? "AUTOPILOT" : "KEYBOARD");
            _sb.AppendFormat(c, "Speed {0:F1} km/h   steer {1:F1}°   grounded {2}/4\n", t.Speed * 3.6f, a.State.SteerDeg, a.State.GroundedWheels);
            _sb.AppendFormat(c, "Lap {0}   next CP {1}/{2}   progress {3:P1}\n", t.Laps, t.NextCheckpoint, a.Track.CheckpointCount, t.Progress);
            _sb.AppendFormat(c, "Lap time {0:F2}s   last {1:F2}s   best {2:F2}s   timing {3}\n", a.LapTimer.Current, t.LastLapTime, t.BestLapTime, a.Tracker.TimingActive);
            _sb.AppendFormat(c, "Signals: wall={0} flip={1} oob={2} stuck={3} wrongHeading={4} nonFinite={5} clearance={6:F2}m\n",
                s.WallContact, s.Flipped, s.OutOfBounds, s.NoProgressTimeout, s.WrongWayHeading, s.NonFinite, s.WallClearance);
            _sb.AppendFormat(c, "Wall contact steps (total): {0}   obs hash {1}\n\n", a.Collisions.TotalWallContactSteps, ObservationSpec.LayoutHash);
            for (int i = 0; i < ObservationSpec.Size; i++)
            {
                _sb.AppendFormat(c, "{0,-10} {1,7:F3}", ObsNames[i], _obs[i]);
                _sb.Append(i % 3 == 2 ? "\n" : "    ");
            }
            GUI.Box(new Rect(10, 10, 560, 330), _sb.ToString(), _style);
        }
    }
}
