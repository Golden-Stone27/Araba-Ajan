using NUnit.Framework;
using Racing.Bridge;
using Racing.Core;

namespace Racing.Tests
{
    /// <summary>M3 wire format: sizes and offsets must match python/racing_rl/bridge/protocol.py.</summary>
    public class BridgeProtocolTests
    {
        [Test]
        public void StateLayout_MatchesContract()
        {
            var l = new BridgeProtocol.StateLayout(16);
            Assert.AreEqual(16 * 26 * 4, l.Reward);
            Assert.AreEqual(l.Reward + 64, l.Terminated);
            Assert.AreEqual(l.Terminated + 16, l.Truncated);
            Assert.AreEqual(l.Truncated + 16, l.FinalObs); // 2N = 32 → no pad
            Assert.AreEqual(4064, l.Size);                 // obs 1664 + rew 64 + flags 32 + final_obs 1664 + info 640
            Assert.AreEqual(2, BridgeProtocol.FlagPad(1));
            Assert.AreEqual(0, BridgeProtocol.FlagPad(2));
            Assert.AreEqual(2, BridgeProtocol.FlagPad(3));
            Assert.AreEqual(0, new BridgeProtocol.StateLayout(1).FinalObs % 4);
            Assert.AreEqual(0, new BridgeProtocol.StateLayout(3).FinalObs % 4);
            Assert.AreEqual(128, BridgeProtocol.StepPayloadSize(16));
        }

        [Test]
        public void Header_RoundTrip()
        {
            var buf = new byte[16];
            BridgeProtocol.WriteHeader(buf, BridgeProtocol.MsgState, 123456u, 4064);
            Assert.AreEqual((byte)'R', buf[0]);
            Assert.AreEqual((byte)'C', buf[1]);
            Assert.AreEqual((byte)'A', buf[2]);
            Assert.AreEqual((byte)'G', buf[3]);
            BridgeProtocol.Header h = BridgeProtocol.ReadHeader(buf);
            Assert.AreEqual(BridgeProtocol.Magic, h.Magic);
            Assert.AreEqual(BridgeProtocol.Version, h.Version);
            Assert.AreEqual(BridgeProtocol.MsgState, h.MsgType);
            Assert.AreEqual(123456u, h.Seq);
            Assert.AreEqual(4064u, h.PayloadLen);
        }

        [Test]
        public void Info_LayoutMatchesRaceInfoV1()
        {
            var t = new AgentTelemetry
            {
                TermReason = TermReason.Finished, Laps = 3, NextCheckpoint = 7, EpisodeDecisions = 1234,
                LastLapTime = 41.28f, BestLapTime = 41.2f, Speed = 30.5f, Progress = 0.25f, EpisodeReturn = -1.5f,
                PosX = 10f, PosZ = -20f
            };
            var buf = new byte[BridgeProtocol.InfoSize + 4];
            BridgeProtocol.WriteInfo(buf, 4, t, true);
            Assert.AreEqual(8, buf[4]);
            Assert.AreEqual(1, buf[5]);
            Assert.AreEqual(3, buf[6] | (buf[7] << 8));
            Assert.AreEqual(7, buf[8] | (buf[9] << 8));
            Assert.AreEqual(1234, System.BitConverter.ToInt32(buf, 12));
            Assert.AreEqual(41.28f, System.BitConverter.ToSingle(buf, 16));
            Assert.AreEqual(-1.5f, System.BitConverter.ToSingle(buf, 32));
            Assert.AreEqual(-20f, System.BitConverter.ToSingle(buf, 40));
        }
    }
}
