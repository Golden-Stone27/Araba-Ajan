"""Launches and supervises one Unity player process (Windows: taskkill /T /F for cleanup)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


class UnityProcess:
    def __init__(self, exe: str | os.PathLike, port: int, num_agents: int, log_path: str | os.PathLike,
                 extra_args: list[str] | None = None, headless: bool = True):
        self.exe = str(Path(exe).resolve())  # CreateProcess does not resolve relative paths with '/'
        self.port = port
        self.num_agents = num_agents
        self.log_path = Path(log_path)
        self.extra_args = list(extra_args or [])
        self.headless = headless
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        args = [self.exe]
        if self.headless:
            args += ["-batchmode", "-nographics"]
        args += ["-logFile", str(self.log_path), "-bridgePort", str(self.port), "-numAgents", str(self.num_agents)]
        args += self.extra_args
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, creationflags=flags)

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def returncode(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def wait(self, timeout_s: float) -> bool:
        if self.proc is None:
            return True
        try:
            self.proc.wait(timeout=timeout_s)
            return True
        except subprocess.TimeoutExpired:
            return False

    def kill_tree(self) -> None:
        if not self.alive():
            return
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            self.proc.kill()
        self.wait(10.0)

    def tail_log(self, lines: int = 50) -> str:
        """Last lines of the Unity log (the player may still be writing it)."""
        for _ in range(3):
            try:
                text = self.log_path.read_text(encoding="utf-8", errors="replace")
                return "\n".join(text.splitlines()[-lines:])
            except OSError:
                time.sleep(0.1)
        return "<unity log unavailable>"
