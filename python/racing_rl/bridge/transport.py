"""Length-prefixed framing over a TCP socket (Python is the server, Unity the client)."""

from __future__ import annotations

import errno
import socket

from .errors import DesyncError, UnityCrashedError, UnityTimeoutError
from .protocol import HEADER, MAGIC, VERSION

SOCKET_BUFFER = 1 << 20


def configure(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_BUFFER)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER)


def bind_server(host: str, port: int, scan: int = 20) -> tuple[socket.socket, int]:
    """Binds host:port, scanning port+1..port+scan on EADDRINUSE. Returns (listening socket, bound port)."""
    last: OSError | None = None
    for p in range(port, port + scan + 1):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Windows: SO_EXCLUSIVEADDRUSE so two trainers never share a port.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            srv.bind((host, p))
        except OSError as e:
            srv.close()
            if e.errno in (errno.EADDRINUSE, errno.EACCES, 10048, 10013):
                last = e
                continue
            raise
        srv.listen(1)
        return srv, p
    raise OSError(f"no free port in {port}..{port + scan}") from last


class FramedSocket:
    """send(msg_type, payload) / recv() -> (msg_type, seq, memoryview). recv reuses one growing buffer."""

    def __init__(self, sock: socket.socket, timeout_s: float | None):
        configure(sock)
        self.sock = sock
        self.sock.settimeout(timeout_s)
        self._send = bytearray(HEADER.size + 64 * 1024)
        self._recv = bytearray(HEADER.size + 64 * 1024)
        self._hdr = memoryview(bytearray(HEADER.size))
        self.sent = 0
        self.received = 0

    def set_timeout(self, timeout_s: float | None) -> None:
        self.sock.settimeout(timeout_s)

    def send(self, msg_type: int, seq: int, payload=b"") -> None:
        n = len(payload)
        total = HEADER.size + n
        if total > len(self._send):
            self._send = bytearray(total)
        HEADER.pack_into(self._send, 0, MAGIC, VERSION, msg_type, seq, n)
        if n:
            self._send[HEADER.size:total] = payload
        try:
            self.sock.sendall(memoryview(self._send)[:total])
        except socket.timeout as e:
            raise UnityTimeoutError("send timed out") from e
        except (ConnectionError, OSError) as e:
            raise UnityCrashedError(f"send failed: {e}") from e
        self.sent += 1

    def recv(self) -> tuple[int, int, memoryview]:
        self._recv_exactly(self._hdr, HEADER.size)
        magic, version, msg_type, seq, n = HEADER.unpack(self._hdr)
        if magic != MAGIC or version != VERSION:
            raise DesyncError(f"bad header magic=0x{magic:08X} version={version}")
        if n > len(self._recv):
            self._recv = bytearray(n)
        view = memoryview(self._recv)[:n]
        self._recv_exactly(view, n)
        self.received += 1
        return msg_type, seq, view

    def _recv_exactly(self, view: memoryview, n: int) -> None:
        got = 0
        try:
            while got < n:
                k = self.sock.recv_into(view[got:], n - got)
                if k == 0:
                    raise UnityCrashedError("connection closed by Unity")
                got += k
        except socket.timeout as e:
            raise UnityTimeoutError("recv timed out") from e
        except (ConnectionResetError, ConnectionAbortedError) as e:
            raise UnityCrashedError(f"connection reset: {e}") from e

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
