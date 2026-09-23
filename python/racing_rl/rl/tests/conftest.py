"""Shared fixtures: every M4 test runs seeded and with deterministic torch kernels (CPU only)."""

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
