using System;
using System.Buffers.Binary;
using System.Globalization;
using Racing.Core;

namespace Racing.Bridge
{
    /// <summary>
    /// Wire format of the M3 bridge (docs/milestones/M3_bridge_gym.md). Little-endian throughout.
    /// Header (16 B, Python struct "&lt;IHHII"): magic, version, msg_type, seq, payload_len.
    /// </summary>
    public static class BridgeProtocol
    {
        public const uint Magic = 0x47414352; // bytes "RCAG"
        public const ushort Version = 1;
        public const int HeaderSize = 16;
        public const int ObsDim = ObservationSpec.Size;
        public const int ActDim = 2;
        public const int InfoSize = 40; // RACE_INFO_V1 '<BBHHHifffffff'
        public const string InfoStruct = "RACE_INFO_V1";
        public const int ResetPayloadSize = 16; // '<qII'
        public const int SocketBufferSize = 1 << 20;

        public const ushort MsgHello = 0x0001;
        public const ushort MsgConfig = 0x0002;
        public const ushort MsgReady = 0x0003;
        public const ushort MsgReset = 0x0010;
        public const ushort MsgStep = 0x0011;
        public const ushort MsgState = 0x0020;
        public const ushort MsgClose = 0x0030;
        public const ushort MsgError = 0x00FF;

        public const string ErrBadMagic = "BAD_MAGIC";
        public const string ErrBadVersion = "BAD_VERSION";
        public const string ErrHashMismatch = "HASH_MISMATCH";
        public const string ErrBadAction = "BAD_ACTION";
        public const string ErrBadLength = "BAD_LENGTH";
        public const string ErrUnexpectedMsg = "UNEXPECTED_MSG";
        public const string ErrUnknownTrack = "UNKNOWN_TRACK";   // M6: -trackName/-trackIndex unresolvable or conflicting (sent instead of HELLO)
        public const string ErrTrackMismatch = "TRACK_MISMATCH"; // M6: CONFIG expected_track_id != the running track

        public static int FlagPad(int n) => (4 - (2 * n) % 4) % 4;

        /// <summary>STATE layout offsets for N agents (bytes, relative to the payload start).</summary>
        public readonly struct StateLayout
        {
            public readonly int N, Obs, Reward, Terminated, Truncated, FinalObs, Info, Size;

            public StateLayout(int n)
            {
                N = n;
                Obs = 0;
                Reward = Obs + n * ObsDim * 4;
                Terminated = Reward + n * 4;
                Truncated = Terminated + n;
                FinalObs = Truncated + n + FlagPad(n);
                Info = FinalObs + n * ObsDim * 4;
                Size = Info + n * InfoSize;
            }
        }

        public static int StepPayloadSize(int n) => n * ActDim * 4;

        public static void WriteHeader(byte[] buf, ushort msgType, uint seq, int payloadLen)
        {
            var s = buf.AsSpan(0, HeaderSize);
            BinaryPrimitives.WriteUInt32LittleEndian(s, Magic);
            BinaryPrimitives.WriteUInt16LittleEndian(s.Slice(4), Version);
            BinaryPrimitives.WriteUInt16LittleEndian(s.Slice(6), msgType);
            BinaryPrimitives.WriteUInt32LittleEndian(s.Slice(8), seq);
            BinaryPrimitives.WriteUInt32LittleEndian(s.Slice(12), (uint)payloadLen);
        }

        public struct Header
        {
            public uint Magic;
            public ushort Version;
            public ushort MsgType;
            public uint Seq;
            public uint PayloadLen;
        }

        public static Header ReadHeader(byte[] buf)
        {
            ReadOnlySpan<byte> s = buf.AsSpan(0, HeaderSize);
            return new Header
            {
                Magic = BinaryPrimitives.ReadUInt32LittleEndian(s),
                Version = BinaryPrimitives.ReadUInt16LittleEndian(s.Slice(4)),
                MsgType = BinaryPrimitives.ReadUInt16LittleEndian(s.Slice(6)),
                Seq = BinaryPrimitives.ReadUInt32LittleEndian(s.Slice(8)),
                PayloadLen = BinaryPrimitives.ReadUInt32LittleEndian(s.Slice(12))
            };
        }

        public static void WriteFloat(Span<byte> s, float v) => BinaryPrimitives.WriteInt32LittleEndian(s, BitConverter.SingleToInt32Bits(v));
        public static float ReadFloat(ReadOnlySpan<byte> s) => BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(s));

        /// <summary>RACE_INFO_V1 (40 B) at buf[offset]. lapCompleted is OR-ed over the K sub-steps of the block.</summary>
        public static void WriteInfo(byte[] buf, int offset, in AgentTelemetry t, bool lapCompleted)
        {
            var s = buf.AsSpan(offset, InfoSize);
            s[0] = (byte)t.TermReason;
            s[1] = lapCompleted ? (byte)1 : (byte)0;
            BinaryPrimitives.WriteUInt16LittleEndian(s.Slice(2), (ushort)Math.Min(t.Laps, ushort.MaxValue));
            BinaryPrimitives.WriteUInt16LittleEndian(s.Slice(4), (ushort)Math.Min(t.NextCheckpoint, ushort.MaxValue));
            BinaryPrimitives.WriteUInt16LittleEndian(s.Slice(6), 0);
            BinaryPrimitives.WriteInt32LittleEndian(s.Slice(8), t.EpisodeDecisions);
            WriteFloat(s.Slice(12), t.LastLapTime);
            WriteFloat(s.Slice(16), t.BestLapTime);
            WriteFloat(s.Slice(20), t.Speed);
            WriteFloat(s.Slice(24), t.Progress);
            WriteFloat(s.Slice(28), t.EpisodeReturn);
            WriteFloat(s.Slice(32), t.PosX);
            WriteFloat(s.Slice(36), t.PosZ);
        }

        /// <summary>HELLO track fields (M6). Index -1 = procedural track (not in the catalog). Hash = the track part alone.</summary>
        public readonly struct TrackInfo
        {
            public readonly string Id;
            public readonly int Index;
            public readonly float LengthM;
            public readonly int Checkpoints;
            public readonly float HalfWidth;
            public readonly string Hash;

            public TrackInfo(string id, int index, float lengthM, int checkpoints, float halfWidth, string hash)
            {
                Id = id;
                Index = index;
                LengthM = lengthM;
                Checkpoints = checkpoints;
                HalfWidth = halfWidth;
                Hash = hash;
            }

            public static TrackInfo From(TrackDefinition def, int index, ITrack track) =>
                new TrackInfo(def.trackId, index, track.Length, track.CheckpointCount, track.HalfWidth, ConfigHash.Compute(def));
        }

        /// <summary>
        /// HELLO payload. The M3 fields keep their order and format; the M6 track fields are appended, so PROTOCOL/Version stay 1.
        /// </summary>
        public static string HelloJson(string env, string unity, string buildId, int numAgents, string envConfigHash, float fixedDt,
                                       int decisionPeriod, int maxEpisodeDecisions, in TrackInfo track)
        {
            var c = CultureInfo.InvariantCulture;
            return string.Format(c,
                "{{\"protocol\":{0},\"env\":\"{1}\",\"unity\":\"{2}\",\"build_id\":\"{3}\",\"num_agents\":{4},\"obs_dim\":{5},\"act_dim\":{6}," +
                "\"obs_layout_hash\":\"{7}\",\"env_config_hash\":\"{8}\",\"fixed_dt\":{9},\"decision_period\":{10},\"max_episode_decisions\":{11}," +
                "\"info_struct\":\"{12}\",\"track_id\":\"{13}\",\"track_index\":{14},\"track_length_m\":{15},\"track_checkpoints\":{16}," +
                "\"track_half_width\":{17},\"track_hash\":\"{18}\"}}",
                Version, JsonEscape(env), unity, JsonEscape(buildId), numAgents, ObsDim, ActDim,
                ObservationSpec.LayoutHash, envConfigHash, fixedDt.ToString("R", c),
                decisionPeriod, maxEpisodeDecisions, InfoStruct, JsonEscape(track.Id), track.Index, track.LengthM.ToString("R", c),
                track.Checkpoints, track.HalfWidth.ToString("R", c), track.Hash);
        }

        /// <summary>CONFIG payload (JsonUtility). Empty expectations are not checked; a missing strict reads as false.</summary>
        [Serializable]
        public struct ConfigMsg
        {
            public string expected_obs_layout_hash;
            public string expected_env_config_hash;
            public string expected_track_id;
            public bool strict;
        }

        /// <summary>
        /// False with (code, message) when CONFIG expects something else. The track id is checked first (TRACK_MISMATCH), since
        /// another track also changes env_config_hash; then the hashes (HASH_MISMATCH).
        /// </summary>
        public static bool CheckConfig(in ConfigMsg cfg, string obsLayoutHash, string envConfigHash, string trackId,
                                       out string code, out string message)
        {
            if (!string.IsNullOrEmpty(cfg.expected_track_id) && !string.Equals(cfg.expected_track_id, trackId, StringComparison.Ordinal))
            {
                code = ErrTrackMismatch;
                message = $"track_id {trackId} (expected {cfg.expected_track_id}), env_config_hash {envConfigHash}";
                return false;
            }
            bool obsOk = string.IsNullOrEmpty(cfg.expected_obs_layout_hash) || string.Equals(cfg.expected_obs_layout_hash, obsLayoutHash, StringComparison.Ordinal);
            bool envOk = string.IsNullOrEmpty(cfg.expected_env_config_hash) || string.Equals(cfg.expected_env_config_hash, envConfigHash, StringComparison.Ordinal);
            if (!(obsOk && envOk))
            {
                code = ErrHashMismatch;
                message = $"obs_layout_hash {obsLayoutHash} (expected {cfg.expected_obs_layout_hash}), " +
                          $"env_config_hash {envConfigHash} (expected {cfg.expected_env_config_hash})";
                return false;
            }
            code = null;
            message = null;
            return true;
        }

        /// <summary>Minimal JSON string escaping for the handshake/error messages (ASCII payloads).</summary>
        public static string JsonEscape(string v)
        {
            if (string.IsNullOrEmpty(v)) return string.Empty;
            return v.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r");
        }
    }
}
