using System.Collections.Generic;
using System.Globalization;
using Racing.Core;
using UnityEngine;
using UnityEngine.UIElements;

namespace Racing.Viewer
{
    /// <summary>
    /// UI1 track picker (UI Toolkit, top left): catalog tracks and "proc:&lt;seed&gt;" (seed field, random seed), autopilot
    /// toggle, reset, and a line with the loaded track's id, profile, length, width and gate count. Controls lose focus
    /// after use so arrow/WASD driving never navigates the UI; P/R shortcuts are off while the seed field has focus.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class ViewerUI : MonoBehaviour
    {
        public const string ProcChoice = "Usulü (proc)";
        public const string TrackDropdownName = "track-dropdown";
        public const string SeedFieldName = "seed-field";
        public const string InfoLabelName = "info-label";
        public const string ErrorLabelName = "error-label";

        [SerializeField] ViewerController viewer;
        [SerializeField] StyleSheet styleSheet;
        [SerializeField] long initialSeed = 1;

        readonly System.Random _random = new System.Random();
        DropdownField _track;
        VisualElement _procRow;
        TextField _seed;
        Toggle _autopilot;
        Label _info, _error;

        void Start()
        {
            VisualElement root = GetComponent<UIDocument>().rootVisualElement;
            if (styleSheet != null && !root.styleSheets.Contains(styleSheet)) root.styleSheets.Add(styleSheet);
            root.Add(Build());
            viewer.CarSpawned += Refresh;
            if (viewer.Car != null) Refresh();
        }

        void OnDestroy()
        {
            if (viewer != null) viewer.CarSpawned -= Refresh;
        }

        void Update()
        {
            if (_autopilot != null && _autopilot.value != viewer.Autopilot) _autopilot.SetValueWithoutNotify(viewer.Autopilot); // P key
        }

        VisualElement Build()
        {
            var panel = new VisualElement();
            panel.AddToClassList("viewer-panel");
            panel.Add(NewLabel("RaceAgent · Pist Seçici", "viewer-title"));

            var choices = new List<string>();
            foreach (TrackDefinition def in viewer.Catalog.tracks)
                if (def != null) choices.Add(def.trackId);
            choices.Add(ProcChoice);
            _track = new DropdownField("Pist", choices, 0) { name = TrackDropdownName };
            _track.RegisterValueChangedCallback(e => SelectTrack(e.newValue));
            panel.Add(_track);

            _procRow = NewRow();
            _seed = new TextField("Tohum") { name = SeedFieldName, maxLength = 19 };
            _seed.SetValueWithoutNotify(initialSeed.ToString(CultureInfo.InvariantCulture));
            _seed.RegisterValueChangedCallback(e => _seed.SetValueWithoutNotify(DigitsOnly(e.newValue)));
            _seed.RegisterCallback<KeyDownEvent>(e =>
            {
                if (e.keyCode == KeyCode.Return || e.keyCode == KeyCode.KeypadEnter) LoadSeed(_seed.value);
            }, TrickleDown.TrickleDown);
            _seed.RegisterCallback<FocusInEvent>(_ => viewer.ShortcutsEnabled = false);
            _seed.RegisterCallback<FocusOutEvent>(_ => viewer.ShortcutsEnabled = true);
            _procRow.Add(_seed);
            _procRow.Add(new Button(() => LoadSeed(_seed.value)) { text = "Yükle" });
            _procRow.Add(new Button(RandomSeed) { text = "Rastgele" });
            _procRow.style.display = DisplayStyle.None;
            panel.Add(_procRow);

            VisualElement controls = NewRow();
            _autopilot = new Toggle("Otopilot (P)") { value = viewer.Autopilot };
            _autopilot.RegisterValueChangedCallback(e =>
            {
                viewer.Autopilot = e.newValue;
                _autopilot.Blur();
            });
            controls.Add(_autopilot);
            var reset = new Button { text = "Sıfırla (R)" };
            reset.clicked += () =>
            {
                viewer.ResetCar();
                reset.Blur();
            };
            controls.Add(reset);
            panel.Add(controls);

            _info = NewLabel("", "viewer-info");
            _info.name = InfoLabelName;
            panel.Add(_info);
            _error = NewLabel("", "viewer-error");
            _error.name = ErrorLabelName;
            panel.Add(_error);
            ShowError(null);
            panel.Add(NewLabel("Klavye: ok tuşları / WASD · P otopilot · R sıfırla", "viewer-hint"));
            return panel;
        }

        /// <summary>Dropdown choice: a catalog track id loads at once; ProcChoice shows the seed row and loads its seed.</summary>
        public void SelectTrack(string choice)
        {
            bool proc = choice == ProcChoice;
            _procRow.style.display = proc ? DisplayStyle.Flex : DisplayStyle.None;
            if (proc) LoadSeed(_seed.value);
            else Load(choice);
            _track.Blur();
        }

        /// <summary>Loads "proc:&lt;digits&gt;"; a non-numeric or out-of-range seed shows an error and keeps the current track.</summary>
        public void LoadSeed(string digits)
        {
            string name = ProceduralTrackGenerator.Prefix + digits;
            if (!ProceduralTrackGenerator.TryParseName(name, out long _))
            {
                ShowError("Tohum 0 veya pozitif bir tam sayı olmalı (en çok 19 basamak).");
                return;
            }
            Load(name);
        }

        public void RandomSeed()
        {
            string digits = _random.Next(0, 1000000).ToString(CultureInfo.InvariantCulture);
            _seed.SetValueWithoutNotify(digits);
            LoadSeed(digits);
        }

        void Load(string trackName)
        {
            ShowError(viewer.LoadTrack(trackName, out string error) ? null : "Yüklenemedi: " + error);
        }

        /// <summary>Shows the loaded track (CarSpawned) and keeps the dropdown and seed field in sync with it.</summary>
        void Refresh()
        {
            RaceEnvironment env = viewer.Environment;
            TrackDefinition def = env.TrackDef;
            _info.text = string.Format(CultureInfo.InvariantCulture, "{0} · {1} · L {2:F1} m · W {3:F1} m · {4} kapı",
                def.trackId, def.profile, env.Track.Length, def.width, env.Track.CheckpointCount);
            bool proc = viewer.TrackIndex < 0;
            _track.SetValueWithoutNotify(proc ? ProcChoice : def.trackId);
            _procRow.style.display = proc ? DisplayStyle.Flex : DisplayStyle.None;
            if (proc && ProceduralTrackGenerator.TryParseName(def.trackId, out long seed))
                _seed.SetValueWithoutNotify(seed.ToString(CultureInfo.InvariantCulture));
        }

        void ShowError(string message)
        {
            _error.text = message ?? "";
            _error.style.display = message == null ? DisplayStyle.None : DisplayStyle.Flex;
        }

        static string DigitsOnly(string s)
        {
            var chars = new List<char>(s.Length);
            foreach (char ch in s)
                if (ch >= '0' && ch <= '9') chars.Add(ch);
            return chars.Count == s.Length ? s : new string(chars.ToArray());
        }

        static Label NewLabel(string text, string cls)
        {
            var label = new Label(text);
            label.AddToClassList(cls);
            return label;
        }

        static VisualElement NewRow()
        {
            var row = new VisualElement();
            row.AddToClassList("viewer-row");
            return row;
        }
    }
}
