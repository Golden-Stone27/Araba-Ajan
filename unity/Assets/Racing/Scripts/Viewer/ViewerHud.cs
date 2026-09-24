using System.Globalization;
using Racing.Core;
using UnityEngine;
using UnityEngine.UIElements;

namespace Racing.Viewer
{
    /// <summary>
    /// UI1 minimal HUD (UI Toolkit, bottom right): drive mode, speed in km/h and the simulated time since the car was
    /// last placed on the start line (ViewerController.ElapsedSeconds). Labels change only when the shown value does.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class ViewerHud : MonoBehaviour
    {
        public const string SpeedLabelName = "hud-speed";
        public const string TimeLabelName = "hud-time";
        public const string ModeLabelName = "hud-mode";

        [SerializeField] ViewerController viewer;
        [SerializeField] StyleSheet styleSheet;

        Label _mode, _speed, _time;
        int _shownSpeed = -1, _shownCentis = -1, _shownAutopilot = -1;

        void Start()
        {
            VisualElement root = GetComponent<UIDocument>().rootVisualElement;
            if (styleSheet != null && !root.styleSheets.Contains(styleSheet)) root.styleSheets.Add(styleSheet);

            var panel = new VisualElement();
            panel.AddToClassList("hud-panel");
            _mode = NewLabel(ModeLabelName, "hud-mode");
            panel.Add(_mode);
            var speedRow = new VisualElement();
            speedRow.AddToClassList("hud-speed-row");
            _speed = NewLabel(SpeedLabelName, "hud-speed");
            speedRow.Add(_speed);
            speedRow.Add(NewLabel(null, "hud-unit", "km/s"));
            panel.Add(speedRow);
            _time = NewLabel(TimeLabelName, "hud-time");
            panel.Add(_time);
            root.Add(panel);
        }

        void LateUpdate()
        {
            if (_speed == null) return;
            RaceAgentCore car = viewer.Car;
            int speed = car != null ? Mathf.RoundToInt(car.Telemetry.Speed * 3.6f) : 0;
            int centis = Mathf.FloorToInt(viewer.ElapsedSeconds * 100f);
            int autopilot = viewer.Autopilot ? 1 : 0;
            if (speed != _shownSpeed) _speed.text = (_shownSpeed = speed).ToString(CultureInfo.InvariantCulture);
            if (centis != _shownCentis) _time.text = "Süre " + FormatCentis(_shownCentis = centis);
            if (autopilot != _shownAutopilot)
            {
                _shownAutopilot = autopilot;
                _mode.text = autopilot == 1 ? "OTOPİLOT" : "KLAVYE";
                _mode.EnableInClassList("hud-mode--manual", autopilot == 0);
            }
        }

        /// <summary>Hundredths of a second as mm:ss.ff (minutes keep counting past 59).</summary>
        public static string FormatCentis(int centis)
        {
            if (centis < 0) centis = 0;
            return string.Format(CultureInfo.InvariantCulture, "{0:00}:{1:00}.{2:00}", centis / 6000, centis / 100 % 60, centis % 100);
        }

        static Label NewLabel(string name, string cls, string text = "")
        {
            var label = new Label(text) { name = name };
            label.AddToClassList(cls);
            return label;
        }
    }
}
