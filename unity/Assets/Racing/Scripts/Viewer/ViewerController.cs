using System;
using System.Collections;
using Racing.Core;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Racing.Viewer
{
    /// <summary>
    /// UI1 track viewer: switches tracks at runtime without reloading the scene and drives one car with PurePursuit
    /// (default) or the keyboard. Every load destroys the previous RaceEnvironment and instantiates a fresh one from an
    /// inactive scene template (configs + materials, initializeOnAwake off), so the track meshes, colliders and the car
    /// are all rebuilt; the car spawns on the EvalGrid start (1 m past the line, fixed seed). Step order is C0.8
    /// (ApplyAction → PhysicsStep), as in SimulationDriver.
    /// Not a measuring tool (C0.21): graded tracks are not bitwise reproducible when rebuilt in one process (C0.20).
    /// Keys: P autopilot, R reset.
    /// </summary>
    [DefaultExecutionOrder(-800)]
    public sealed class ViewerController : MonoBehaviour
    {
        [Tooltip("Inactive RaceEnvironment with configs and materials; initializeOnAwake must be off.")]
        [SerializeField] RaceEnvironment template;
        [SerializeField] TrackCatalog catalog;
        [SerializeField] string initialTrack = TrackCatalog.BenchmarkId;
        [SerializeField] long spawnSeed = 1000;
        [SerializeField] bool autopilot = true;

        readonly KeyboardInputSource _keyboard = new KeyboardInputSource();
        PurePursuitInputSource _pilot;
        RaceEnvironment _env;
        TrackDefinition _generated; // proc track (DontSave: never unloaded automatically), destroyed on the next load

        /// <summary>Raised after every track load and car reset, once the new car is on the start line.</summary>
        public event Action CarSpawned;

        public TrackCatalog Catalog => catalog;
        public RaceEnvironment Environment => _env;
        public RaceAgentCore Car => _env != null && _env.IsInitialized ? _env.Agents[0] : null;
        public string TrackName { get; private set; }
        /// <summary>Catalog index of the loaded track; -1 for proc tracks.</summary>
        public int TrackIndex { get; private set; } = -1;
        public int LoadCount { get; private set; }
        public bool ShortcutsEnabled { get; set; } = true;

        public bool Autopilot
        {
            get => autopilot;
            set => autopilot = value;
        }

        /// <summary>Simulated seconds since the car was last placed on the start line.</summary>
        public float ElapsedSeconds => Car != null ? Car.PhysicsSteps * _env.Sim.fixedDeltaTime : 0f;

        void Start()
        {
            if (!LoadTrack(initialTrack, out string error)) Debug.LogError("[Viewer] " + error);
        }

        void Update()
        {
            if (!ShortcutsEnabled) return;
            if (Input.GetKeyDown(KeyCode.P)) autopilot = !autopilot;
            if (Input.GetKeyDown(KeyCode.R)) ResetCar();
        }

        void FixedUpdate()
        {
            RaceAgentCore car = Car;
            if (car == null) return;
            car.ApplyAction(autopilot ? _pilot.ReadAction() : _keyboard.ReadAction());
            AgentStepResult r = _env.PhysicsStep()[0];
            EpisodeSignals s = r.Signals;
            // The time limit (truncation) is ignored so the autopilot keeps lapping. A crash restarts the autopilot car;
            // the keyboard car restarts only on flip / out-of-bounds / non-finite, as in the Sandbox.
            if (s.Flipped || s.OutOfBounds || s.NonFinite || (autopilot && r.Terminated)) _env.ResetAgent(0);
        }

        void OnDestroy()
        {
            if (_generated != null) Destroy(_generated);
        }

        /// <summary>
        /// Resolves a catalog id / asset name or "proc:&lt;seed&gt;" (TrackCatalog.TryResolve, as the bridge does) and
        /// rebuilds the environment on it. On failure the current track stays loaded.
        /// </summary>
        public bool LoadTrack(string trackName, out string error)
        {
            error = null;
            if (template == null || catalog == null)
            {
                error = "viewer template or track catalog is not assigned";
                return false;
            }
            TrackDefinition def;
            int index;
            try
            {
                if (!catalog.TryResolve(trackName, out def, out index))
                {
                    error = "unknown track: " + trackName;
                    return false;
                }
            }
            catch (Exception e) // proc generation can fail (no valid candidate within MaxAttempts)
            {
                error = trackName + ": " + e.Message;
                return false;
            }

            DestroyEnvironment();
            GameObject go = Instantiate(template.gameObject); // the template is inactive, so is the clone: no Awake yet
            go.name = "RaceEnvironment [" + trackName + "]";
            SceneManager.MoveGameObjectToScene(go, gameObject.scene); // Instantiate uses the active scene
            go.SetActive(true);
            _env = go.GetComponent<RaceEnvironment>();
            _env.InitializeFromSerialized(1, spawnSeed, StartMode.EvalGrid, def);
            _generated = index < 0 ? def : null;
            TrackName = trackName;
            TrackIndex = index;
            LoadCount++;
            OnCarSpawned();
            StartCoroutine(UnloadUnusedAssetsNextFrame());
            return true;
        }

        /// <summary>Fresh car on the start line. RebuildAgents, because a teleport keeps the WheelCollider state (C0.17).</summary>
        public void ResetCar()
        {
            if (Car == null) return;
            _env.RebuildAgents();
            OnCarSpawned();
        }

        void OnCarSpawned()
        {
            _pilot = new PurePursuitInputSource(_env.Agents[0]);
            CarSpawned?.Invoke();
        }

        void DestroyEnvironment()
        {
            if (_env != null)
            {
                GameObject go = _env.gameObject;
                go.SetActive(false); // leaves the physics scene now; Destroy completes at the end of the frame
                Destroy(go);
            }
            if (_generated != null) Destroy(_generated);
            _env = null;
            _generated = null;
            _pilot = null;
        }

        /// <summary>Frees the replaced track's runtime meshes and materials once its objects are gone.</summary>
        static IEnumerator UnloadUnusedAssetsNextFrame()
        {
            yield return null;
            Resources.UnloadUnusedAssets();
        }
    }
}
