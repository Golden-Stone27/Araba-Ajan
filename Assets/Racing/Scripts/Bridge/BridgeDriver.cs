using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Text;
using Racing.Core;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Racing.Bridge
{
    /// <summary>
    /// M3 driver: Unity is the TCP client of a Python server and never steps physics without a STEP.
    /// STEP (C0.8, M3 order): for k in 0..K-1 { ApplyAction(done_i ? Zero : a_i) for every i; PhysicsStep();
    /// accumulate reward; on done capture final_obs/final_info, ResetAgent(i), done_i = true } → STATE.
    /// Rewards earned by a reset agent while it waits for the block to end belong to its new episode and are
    /// carried into the next STATE, so Σ reward over an episode equals RACE_INFO.ep_return.
    /// Steady-state STEP handling is allocation-free (pre-allocated buffers).
    /// Args: -bridgePort p (6005), -numAgents N, -bridgeTimingLog path (per-request Unity processing time).
    /// </summary>
    [DefaultExecutionOrder(-1000)]
    public sealed class BridgeDriver : MonoBehaviour
    {
        public const int DefaultPort = 6005;
        const int GcWarmupSteps = 100;
        const double EditorBudgetMs = 50.0;

        [SerializeField] RaceEnvironment environment;
        [SerializeField] int port = DefaultPort;
        [SerializeField] int numAgents = 16;
        [SerializeField] string host = "127.0.0.1";

        BridgeClient _client;
        BridgeProtocol.StateLayout _layout;
        int _n, _k;
        float[] _actions, _obs, _finalObs;
        double[] _reward, _carry;
        bool[] _done, _term, _trunc, _lap;
        AgentTelemetry[] _finalInfo;
        bool _running;

        // Diagnostics
        long _requests, _steps, _resets, _doubleDone;
        long _gcSteadyBytes, _gcSteadySteps;
        string _timingPath;
        uint[] _timingSeq;
        ushort[] _timingType;
        float[] _timingUs;
        int _timingCount;

        public RaceEnvironment Environment => environment;

        void Awake()
        {
            CultureBootstrap.Apply();
            Application.runInBackground = true;
            Application.targetFrameRate = -1;
            QualitySettings.vSyncCount = 0;
            Physics.simulationMode = SimulationMode.Script;

            port = CommandLineArgs.GetInt("-bridgePort", port);
            numAgents = Mathf.Max(1, CommandLineArgs.GetInt("-numAgents", numAgents));
            _timingPath = CommandLineArgs.GetString("-bridgeTimingLog");

            if (environment == null) environment = FindAnyObjectByType<RaceEnvironment>();
            environment.InitializeFromSerialized(numAgents, 0, StartMode.TrainRandom);
            Allocate(environment.Agents.Count, environment.Sim.decisionPeriod);

            try
            {
                _client.Connect(host, port);
                _running = Handshake();
            }
            catch (Exception e) when (e is IOException || e is SocketException)
            {
                Fail("handshake: " + e.Message);
            }
        }

        void Allocate(int n, int k)
        {
            _n = n;
            _k = Mathf.Max(1, k);
            _layout = new BridgeProtocol.StateLayout(n);
            int cap = Mathf.Max(_layout.Size, BridgeProtocol.StepPayloadSize(n)) + 64 * 1024; // + room for handshake JSON
            _client = new BridgeClient(cap);
            _actions = new float[n * BridgeProtocol.ActDim];
            _obs = new float[n * BridgeProtocol.ObsDim];
            _finalObs = new float[n * BridgeProtocol.ObsDim];
            _reward = new double[n];
            _carry = new double[n];
            _done = new bool[n];
            _term = new bool[n];
            _trunc = new bool[n];
            _lap = new bool[n];
            _finalInfo = new AgentTelemetry[n];
            if (!string.IsNullOrEmpty(_timingPath))
            {
                const int cap2 = 1 << 21;
                _timingSeq = new uint[cap2];
                _timingType = new ushort[cap2];
                _timingUs = new float[cap2];
            }
        }

        [Serializable]
        struct ConfigMsg
        {
            public string expected_obs_layout_hash;
            public string expected_env_config_hash;
            public bool strict;
        }

        bool Handshake()
        {
            SimConfig sim = environment.Sim;
            string build = string.IsNullOrEmpty(Application.buildGUID) ? "editor" : Application.buildGUID;
            string hello = string.Format(CultureInfo.InvariantCulture,
                "{{\"protocol\":{0},\"env\":\"{1}\",\"unity\":\"{2}\",\"build_id\":\"{3}\",\"num_agents\":{4},\"obs_dim\":{5},\"act_dim\":{6}," +
                "\"obs_layout_hash\":\"{7}\",\"env_config_hash\":\"{8}\",\"fixed_dt\":{9},\"decision_period\":{10},\"max_episode_decisions\":{11}," +
                "\"info_struct\":\"{12}\"}}",
                BridgeProtocol.Version, BridgeProtocol.JsonEscape(environment.TrackDef != null ? environment.TrackDef.name : "track"),
                Application.unityVersion, BridgeProtocol.JsonEscape(build), _n, BridgeProtocol.ObsDim, BridgeProtocol.ActDim,
                ObservationSpec.LayoutHash, environment.EnvConfigHash, sim.fixedDeltaTime.ToString("R", CultureInfo.InvariantCulture),
                _k, sim.maxEpisodeDecisions, BridgeProtocol.InfoStruct);
            _client.SendJson(BridgeProtocol.MsgHello, 0, hello);

            BridgeProtocol.Header h = _client.Receive();
            if (!CheckHeader(h)) return false;
            if (h.MsgType != BridgeProtocol.MsgConfig)
            {
                SendError(h.Seq, BridgeProtocol.ErrUnexpectedMsg, "expected CONFIG, got 0x" + h.MsgType.ToString("X4"), true);
                return false;
            }
            var cfg = JsonUtility.FromJson<ConfigMsg>(_client.PayloadString(h));
            bool obsOk = string.IsNullOrEmpty(cfg.expected_obs_layout_hash) || string.Equals(cfg.expected_obs_layout_hash, ObservationSpec.LayoutHash, StringComparison.Ordinal);
            bool envOk = string.IsNullOrEmpty(cfg.expected_env_config_hash) || string.Equals(cfg.expected_env_config_hash, environment.EnvConfigHash, StringComparison.Ordinal);
            if (!(obsOk && envOk))
            {
                string msg = $"obs_layout_hash {ObservationSpec.LayoutHash} (expected {cfg.expected_obs_layout_hash}), " +
                             $"env_config_hash {environment.EnvConfigHash} (expected {cfg.expected_env_config_hash})";
                if (cfg.strict)
                {
                    SendError(h.Seq, BridgeProtocol.ErrHashMismatch, msg, true);
                    return false;
                }
                Debug.LogWarning("[Bridge] hash mismatch (strict=false): " + msg);
            }
            _client.SendJson(BridgeProtocol.MsgReady, h.Seq, "{\"ok\":true}");
            Debug.Log(string.Format(CultureInfo.InvariantCulture, "[Bridge] READY agents={0} K={1} state_bytes={2} env_config_hash={3} obs_layout_hash={4} timing_log={5}",
                _n, _k, _layout.Size, environment.EnvConfigHash, ObservationSpec.LayoutHash, _timingPath ?? "-"));
            return true;
        }

        void Update()
        {
            if (!_running) return;
            try
            {
                if (Application.isBatchMode)
                {
                    HandleOne();
                    return;
                }
                long start = Stopwatch.GetTimestamp();
                while (_running && (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency < EditorBudgetMs)
                    HandleOne();
            }
            catch (Exception e) when (e is IOException || e is SocketException || e is ObjectDisposedException)
            {
                Fail(e.Message);
            }
        }

        void HandleOne()
        {
            BridgeProtocol.Header h = _client.Receive();
            long t0 = Stopwatch.GetTimestamp();
            long gc0 = GC.GetAllocatedBytesForCurrentThread();
            _requests++;
            if (!CheckHeader(h)) return;

            switch (h.MsgType)
            {
                case BridgeProtocol.MsgStep:
                    if (h.PayloadLen != BridgeProtocol.StepPayloadSize(_n))
                    {
                        SendError(h.Seq, BridgeProtocol.ErrBadLength, $"STEP payload {h.PayloadLen} != {BridgeProtocol.StepPayloadSize(_n)}", false);
                        break;
                    }
                    Buffer.BlockCopy(_client.RecvBuffer, BridgeProtocol.HeaderSize, _actions, 0, _actions.Length * 4);
                    if (!ActionsFinite())
                    {
                        SendError(h.Seq, BridgeProtocol.ErrBadAction, "non-finite action", false);
                        break;
                    }
                    StepBlock();
                    SendState(h.Seq);
                    _steps++;
                    if (_steps > GcWarmupSteps)
                    {
                        _gcSteadyBytes += GC.GetAllocatedBytesForCurrentThread() - gc0;
                        _gcSteadySteps++;
                    }
                    break;
                case BridgeProtocol.MsgReset:
                    if (h.PayloadLen != BridgeProtocol.ResetPayloadSize)
                    {
                        SendError(h.Seq, BridgeProtocol.ErrBadLength, $"RESET payload {h.PayloadLen} != {BridgeProtocol.ResetPayloadSize}", false);
                        break;
                    }
                    ResetEnvironment();
                    SendState(h.Seq);
                    _resets++;
                    break;
                case BridgeProtocol.MsgClose:
                    Close("CLOSE received", 0);
                    return;
                case BridgeProtocol.MsgError:
                    Debug.LogError("[Bridge] ERROR from Python: " + _client.PayloadString(h));
                    Close("ERROR received", 2);
                    return;
                default:
                    SendError(h.Seq, BridgeProtocol.ErrUnexpectedMsg, "unexpected msg_type 0x" + h.MsgType.ToString("X4"), false);
                    break;
            }
            RecordTiming(h, t0);
        }

        bool CheckHeader(in BridgeProtocol.Header h)
        {
            if (h.Magic != BridgeProtocol.Magic)
            {
                SendError(h.Seq, BridgeProtocol.ErrBadMagic, "bad magic 0x" + h.Magic.ToString("X8"), true);
                Close("bad magic", 2);
                return false;
            }
            if (h.Version != BridgeProtocol.Version)
            {
                SendError(h.Seq, BridgeProtocol.ErrBadVersion, "version " + h.Version, true);
                Close("bad version", 2);
                return false;
            }
            return true;
        }

        bool ActionsFinite()
        {
            for (int i = 0; i < _actions.Length; i++)
                if (!float.IsFinite(_actions[i])) return false;
            return true;
        }

        void ResetEnvironment()
        {
            var p = _client.RecvBuffer.AsSpan(BridgeProtocol.HeaderSize, BridgeProtocol.ResetPayloadSize);
            long seed = System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(p);
            uint mode = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(p.Slice(8));
            uint maxLaps = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(p.Slice(12));
            // Fresh cars: WheelCollider state survives a teleport, so RESET(seed) is only reproducible with new cars.
            // Auto-resets inside STEP keep the Core teleport (same as M2 training).
            environment.RebuildAgents();
            environment.ResetAll(seed, (StartMode)mode, (int)maxLaps);
            for (int i = 0; i < _n; i++)
            {
                _reward[i] = 0;
                _carry[i] = 0;
                _done[i] = _term[i] = _trunc[i] = _lap[i] = false;
            }
            Array.Clear(_finalObs, 0, _finalObs.Length);
        }

        void StepBlock()
        {
            var agents = environment.Agents;
            for (int i = 0; i < _n; i++)
            {
                _reward[i] = _carry[i];
                _carry[i] = 0;
                _done[i] = _term[i] = _trunc[i] = _lap[i] = false;
            }
            Array.Clear(_finalObs, 0, _finalObs.Length);

            for (int k = 0; k < _k; k++)
            {
                for (int i = 0; i < _n; i++)
                    agents[i].ApplyAction(_done[i] ? VehicleAction.Zero : new VehicleAction(_actions[2 * i], _actions[2 * i + 1]));
                AgentStepResult[] results = environment.PhysicsStep();
                for (int i = 0; i < _n; i++)
                {
                    AgentStepResult r = results[i];
                    bool ended = r.Terminated || r.Truncated;
                    if (_done[i])
                    {
                        // New episode waiting for the block to end (zero action): its reward starts the next STATE.
                        _carry[i] += r.Reward;
                        if (ended)
                        {
                            _doubleDone++;
                            _carry[i] = 0;
                            environment.ResetAgent(i);
                        }
                        continue;
                    }
                    _reward[i] += r.Reward;
                    if (r.LapCompleted) _lap[i] = true;
                    if (!ended) continue;
                    _done[i] = true;
                    _term[i] = r.Terminated;
                    _trunc[i] = r.Truncated;
                    agents[i].WriteObservation(_finalObs, i * BridgeProtocol.ObsDim);
                    _finalInfo[i] = agents[i].Telemetry;
                    environment.ResetAgent(i);
                }
            }
        }

        void SendState(uint seq)
        {
            var agents = environment.Agents;
            byte[] buf = _client.SendBuffer;
            int o = BridgeProtocol.HeaderSize;
            for (int i = 0; i < _n; i++) agents[i].WriteObservation(_obs, i * BridgeProtocol.ObsDim);
            Buffer.BlockCopy(_obs, 0, buf, o + _layout.Obs, _obs.Length * 4);
            for (int i = 0; i < _n; i++)
            {
                BridgeProtocol.WriteFloat(buf.AsSpan(o + _layout.Reward + 4 * i), (float)_reward[i]);
                buf[o + _layout.Terminated + i] = _term[i] ? (byte)1 : (byte)0;
                buf[o + _layout.Truncated + i] = _trunc[i] ? (byte)1 : (byte)0;
                if (_done[i]) BridgeProtocol.WriteInfo(buf, o + _layout.Info + i * BridgeProtocol.InfoSize, _finalInfo[i], _lap[i]);
                else BridgeProtocol.WriteInfo(buf, o + _layout.Info + i * BridgeProtocol.InfoSize, agents[i].Telemetry, _lap[i]);
            }
            for (int p = _layout.Truncated + _n; p < _layout.FinalObs; p++) buf[o + p] = 0;
            Buffer.BlockCopy(_finalObs, 0, buf, o + _layout.FinalObs, _finalObs.Length * 4);
            _client.Send(BridgeProtocol.MsgState, seq, _layout.Size);
        }

        void SendError(uint seq, string code, string message, bool fatal)
        {
            Debug.LogError($"[Bridge] ERROR {code}: {message}");
            try
            {
                _client.SendJson(BridgeProtocol.MsgError, seq,
                    $"{{\"code\":\"{code}\",\"message\":\"{BridgeProtocol.JsonEscape(message)}\",\"fatal\":{(fatal ? "true" : "false")}}}");
            }
            catch (Exception e) when (e is IOException || e is SocketException) { }
            if (fatal) Close(code, 2);
        }

        void RecordTiming(in BridgeProtocol.Header h, long t0)
        {
            if (_timingUs == null || _timingCount >= _timingUs.Length) return;
            _timingSeq[_timingCount] = h.Seq;
            _timingType[_timingCount] = h.MsgType;
            _timingUs[_timingCount] = (float)((Stopwatch.GetTimestamp() - t0) * 1e6 / Stopwatch.Frequency);
            _timingCount++;
        }

        void Fail(string reason)
        {
            Debug.LogError("[Bridge] connection lost: " + reason);
            Close(reason, 2);
        }

        void Close(string reason, int exitCode)
        {
            if (!_running && _client == null) return;
            _running = false;
            string stats = string.Format(CultureInfo.InvariantCulture,
                "{{\"reason\":\"{0}\",\"requests\":{1},\"steps\":{2},\"resets\":{3},\"messages_sent\":{4},\"messages_received\":{5}," +
                "\"gc_steady_bytes\":{6},\"gc_steady_steps\":{7},\"double_done\":{8},\"num_agents\":{9}}}",
                BridgeProtocol.JsonEscape(reason), _requests, _steps, _resets, _client?.MessagesSent ?? 0, _client?.MessagesReceived ?? 0,
                _gcSteadyBytes, _gcSteadySteps, _doubleDone, _n);
            Debug.Log("[Bridge] stats " + stats);
            WriteTimingLog(stats);
            _client?.Dispose();
            _client = null;
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit(exitCode);
#endif
        }

        /// <summary>Binary records '&lt;IHxxf' (seq, msg_type, unity_us) + a sidecar JSON with the stats.</summary>
        void WriteTimingLog(string stats)
        {
            if (string.IsNullOrEmpty(_timingPath)) return;
            try
            {
                string dir = Path.GetDirectoryName(Path.GetFullPath(_timingPath));
                if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                using (var w = new BinaryWriter(File.Create(_timingPath)))
                {
                    for (int i = 0; i < _timingCount; i++)
                    {
                        w.Write(_timingSeq[i]);
                        w.Write(_timingType[i]);
                        w.Write((ushort)0);
                        w.Write(_timingUs[i]);
                    }
                }
                File.WriteAllText(_timingPath + ".json", stats + "\n", new UTF8Encoding(false));
            }
            catch (IOException e)
            {
                Debug.LogError("[Bridge] timing log: " + e.Message);
            }
        }

        void OnDestroy()
        {
            if (_running) Close("destroyed", 0);
        }
    }
}
