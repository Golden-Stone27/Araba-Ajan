"""Policies usable over the bridge: obs f32[N, 26] -> actions f32[N, 2]."""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np

from .protocol import ACT_DIM, OBS_DIM

MLAGENTS_INPUT = "obs_0"
MLAGENTS_DETERMINISTIC_OUTPUT = "deterministic_continuous_actions"
MLAGENTS_STOCHASTIC_OUTPUT = "continuous_actions"


class Policy(Protocol):
    def __call__(self, obs: np.ndarray) -> np.ndarray: ...


class ZeroPolicy:
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros((obs.shape[0], ACT_DIM), np.float32)


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, (obs.shape[0], ACT_DIM)).astype(np.float32)


class OnnxPolicy:
    """
    ML-Agents ONNX export (C0.10: deterministic → μ). Input/output names are checked with onnx.load before
    the session is created. Output is used as-is (ML-Agents clips inside the graph); Unity clips again (C0.5).
    """

    def __init__(self, path: str | os.PathLike, deterministic: bool = True, threads: int = 1):
        import onnx
        import onnxruntime as ort

        self.path = str(path)
        model = onnx.load(self.path)
        inputs = [i.name for i in model.graph.input]
        outputs = [o.name for o in model.graph.output]
        self.output = MLAGENTS_DETERMINISTIC_OUTPUT if deterministic else MLAGENTS_STOCHASTIC_OUTPUT
        if MLAGENTS_INPUT not in inputs:
            raise ValueError(f"{self.path}: expected input '{MLAGENTS_INPUT}', found {inputs}")
        if self.output not in outputs:
            raise ValueError(f"{self.path}: expected output '{self.output}', found {outputs}")
        self.input_names = inputs
        self.output_names = outputs
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(self.path, sess_options=opts, providers=["CPUExecutionProvider"])
        shape = self.session.get_inputs()[[i.name for i in self.session.get_inputs()].index(MLAGENTS_INPUT)].shape
        if shape[-1] not in (OBS_DIM, None) and not isinstance(shape[-1], str):
            raise ValueError(f"{self.path}: obs_0 has {shape[-1]} features, expected {OBS_DIM}")
        meta = [n for n in ("version_number", "memory_size", "continuous_action_output_shape") if n in outputs]
        values = self.session.run(meta, {MLAGENTS_INPUT: np.zeros((1, OBS_DIM), np.float32)}) if meta else []
        self.metadata = {n: np.asarray(v).ravel().tolist() for n, v in zip(meta, values)}

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self.session.run([self.output], {MLAGENTS_INPUT: np.ascontiguousarray(obs, np.float32)})[0]
        return np.asarray(out, np.float32).reshape(obs.shape[0], ACT_DIM)

