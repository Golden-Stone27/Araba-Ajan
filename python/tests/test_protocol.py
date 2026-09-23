import struct

import numpy as np

from racing_rl.bridge.protocol import (FROZEN_OBS_LAYOUT_HASH, HEADER, OBS_DIM, RACE_INFO_DTYPE, RESET, StateLayout,
                                       decode_state, encode_state, flag_pad, layout_hash)


def test_struct_sizes():
    assert struct.calcsize("<IHHII") == 16 == HEADER.size
    assert RACE_INFO_DTYPE.itemsize == 40
    assert RESET.size == 16


def test_layout_hash_matches_frozen():
    assert layout_hash() == FROZEN_OBS_LAYOUT_HASH


def test_state_layout_n16():
    lay = StateLayout(16)
    assert lay.reward == 16 * OBS_DIM * 4
    assert lay.final_obs == lay.truncated + 16  # 2N = 32 → no pad
    assert lay.size == 4064  # doc's "4448" is an arithmetic slip; field-by-field layout gives 4064


def test_pad_keeps_final_obs_aligned():
    for n in range(1, 40):
        assert StateLayout(n).final_obs % 4 == 0
        assert flag_pad(n) in (0, 2)


def test_info_field_offsets():
    offs = {name: RACE_INFO_DTYPE.fields[name][1] for name in RACE_INFO_DTYPE.names}
    assert offs == {"term_reason": 0, "lap_completed": 1, "laps": 2, "next_cp": 4, "reserved": 6, "ep_decisions": 8,
                    "last_lap_s": 12, "best_lap_s": 16, "speed_mps": 20, "progress": 24, "ep_return": 28,
                    "pos_x": 32, "pos_z": 36}


def test_state_roundtrip():
    rng = np.random.default_rng(0)
    for n in (1, 3, 16):
        obs = rng.standard_normal((n, OBS_DIM)).astype(np.float32)
        fin = rng.standard_normal((n, OBS_DIM)).astype(np.float32)
        rew = rng.standard_normal(n).astype(np.float32)
        term = rng.integers(0, 2, n).astype(np.uint8)
        trunc = rng.integers(0, 2, n).astype(np.uint8)
        info = np.zeros(n, RACE_INFO_DTYPE)
        info["laps"] = np.arange(n)
        info["ep_return"] = rew
        buf = encode_state(obs, rew, term, trunc, fin, info)
        sv = decode_state(memoryview(buf), n)
        assert np.array_equal(sv.obs, obs) and np.array_equal(sv.final_obs, fin)
        assert np.array_equal(sv.reward, rew) and np.array_equal(sv.terminated, term) and np.array_equal(sv.truncated, trunc)
        assert np.array_equal(sv.info, info)
        assert sv.obs.base is not None  # zero-copy view
