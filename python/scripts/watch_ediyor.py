"""best.pt'yi Editor'deki Race_BRIDGE sahnesinde gerçek zamanlı sürdürür. Ctrl+C ile durur."""
import sys, time
import numpy as np, torch
from racing_rl.bridge.protocol import StartMode, FROZEN_ENV_CONFIG_HASH, FROZEN_OBS_LAYOUT_HASH
from racing_rl.bridge.vec_env import UnityVecEnv
from racing_rl.rl import TorchPolicy

ckpt = sys.argv[1] if len(sys.argv) > 1 else r"..\benchmarks\models\custom_ppo_s2.pt"
torch.set_num_threads(1)
pol = TorchPolicy.from_checkpoint(ckpt, deterministic=True, expected_env_hash=FROZEN_ENV_CONFIG_HASH,
                                  expected_obs_hash=FROZEN_OBS_LAYOUT_HASH)
print("Port 6005 dinleniyor. Şimdi Unity'de Race_Bridge sahnesinde Play'e basın (180 s süre var)...")
env = UnityVecEnv(None, port=6005)          # exe_path=None: Editor'ün bağlanmasını bekler
dt = env.hello["fixed_dt"] * env.hello["decision_period"]   # 0.1 s / karar
obs, info = env.reset(seed=1000, options={"start_mode": int(StartMode.EVAL_GRID), "max_laps": 3})
try:
    while True:
        t0 = time.perf_counter()
        obs, _, term, trunc, info = env.step(np.asarray(pol(obs), np.float32))
        if info["lap_completed"][0]:
            print(f"Araç 0 tur: {float(info['last_lap_s'][0]):.2f} s")
        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))   # gerçek zamana kilitle
except KeyboardInterrupt:
    pass
finally:
    env.close()