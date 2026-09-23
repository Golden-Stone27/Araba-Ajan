"""Bridge exception hierarchy (M3 error-handling matrix)."""


class BridgeError(RuntimeError):
    """Base class for every bridge failure."""


class UnityLaunchError(BridgeError):
    """Unity could not be started or did not connect within launch_timeout_s."""


class UnityTimeoutError(BridgeError):
    """No reply within step_timeout_s. The process has been killed."""


class UnityCrashedError(BridgeError):
    """Connection closed/reset by the Unity side."""


class DesyncError(BridgeError):
    """Reply seq/type does not match the request (fatal)."""


class ProtocolMismatchError(BridgeError):
    """Handshake mismatch: protocol, dimensions or hashes differ."""


class RemoteError(BridgeError):
    """Unity answered a request with ERROR {code, message, fatal}."""

    def __init__(self, code: str, message: str, fatal: bool):
        super().__init__(f"{code}: {message}{' (fatal)' if fatal else ''}")
        self.code = code
        self.fatal = fatal
