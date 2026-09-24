using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace Racing.Bridge
{
    /// <summary>
    /// Blocking TCP client (Unity side). One send and one receive buffer, allocated once; the payload is written
    /// in place after the 16-byte header so every message goes out with a single Send call.
    /// </summary>
    public sealed class BridgeClient : IDisposable
    {
        public const int ConnectAttempts = 30;
        public const int ReadTimeoutMs = 120_000;

        Socket _socket;
        readonly byte[] _send;
        readonly byte[] _recv;

        public byte[] SendBuffer => _send;
        public byte[] RecvBuffer => _recv;
        public int PayloadOffset => BridgeProtocol.HeaderSize;
        public long MessagesSent { get; private set; }
        public long MessagesReceived { get; private set; }

        public BridgeClient(int capacity)
        {
            _send = new byte[BridgeProtocol.HeaderSize + capacity];
            _recv = new byte[BridgeProtocol.HeaderSize + capacity];
        }

        /// <summary>30 × 1 s retries (Python binds first, then launches Unity; the Editor may be started before Python).</summary>
        public void Connect(string host, int port)
        {
            Exception last = null;
            for (int attempt = 1; attempt <= ConnectAttempts; attempt++)
            {
                var s = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp)
                {
                    NoDelay = true,
                    SendBufferSize = BridgeProtocol.SocketBufferSize,
                    ReceiveBufferSize = BridgeProtocol.SocketBufferSize,
                    ReceiveTimeout = ReadTimeoutMs,
                    SendTimeout = ReadTimeoutMs
                };
                try
                {
                    s.Connect(host, port);
                    _socket = s;
                    Debug.Log($"[Bridge] Connected to {host}:{port} (attempt {attempt})");
                    return;
                }
                catch (SocketException e)
                {
                    last = e;
                    s.Dispose();
                    Thread.Sleep(1000);
                }
            }
            throw new IOException($"Bridge: could not connect to {host}:{port} after {ConnectAttempts} attempts", last);
        }

        /// <summary>Sends header + payload already written at SendBuffer[16..16+payloadLen).</summary>
        public void Send(ushort msgType, uint seq, int payloadLen)
        {
            BridgeProtocol.WriteHeader(_send, msgType, seq, payloadLen);
            int total = BridgeProtocol.HeaderSize + payloadLen;
            int sent = 0;
            while (sent < total)
            {
                int n = _socket.Send(_send, sent, total - sent, SocketFlags.None);
                if (n <= 0) throw new IOException("Bridge: socket closed while sending");
                sent += n;
            }
            MessagesSent++;
        }

        public void SendJson(ushort msgType, uint seq, string json)
        {
            int len = Encoding.UTF8.GetBytes(json, 0, json.Length, _send, BridgeProtocol.HeaderSize);
            Send(msgType, seq, len);
        }

        /// <summary>Blocking read of one message; the payload lands at RecvBuffer[16..].</summary>
        public BridgeProtocol.Header Receive()
        {
            ReadExactly(0, BridgeProtocol.HeaderSize);
            BridgeProtocol.Header h = BridgeProtocol.ReadHeader(_recv);
            if (h.Magic == BridgeProtocol.Magic && h.PayloadLen <= (uint)(_recv.Length - BridgeProtocol.HeaderSize))
                ReadExactly(BridgeProtocol.HeaderSize, (int)h.PayloadLen);
            MessagesReceived++;
            return h;
        }

        public string PayloadString(in BridgeProtocol.Header h) =>
            Encoding.UTF8.GetString(_recv, BridgeProtocol.HeaderSize, (int)h.PayloadLen);

        void ReadExactly(int offset, int count)
        {
            while (count > 0)
            {
                int n = _socket.Receive(_recv, offset, count, SocketFlags.None);
                if (n <= 0) throw new IOException("Bridge: connection closed by peer");
                offset += n;
                count -= n;
            }
        }

        public void Dispose()
        {
            if (_socket == null) return;
            try { _socket.Shutdown(SocketShutdown.Both); } catch (SocketException) { } catch (ObjectDisposedException) { }
            _socket.Dispose();
            _socket = null;
        }
    }
}
