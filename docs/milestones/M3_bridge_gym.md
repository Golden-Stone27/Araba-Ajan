# M3: Protokol ve Gym Entegrasyonu (C# ve Python köprüsü)

> **Alt ajan talimatı:** Bu aşamayı uygulamak için yalnızca bu dosya ve [`../contracts.md`](../contracts.md) yeterlidir.
> Sözleşmelerdeki (C0.x) bir değeri değiştirmen gerekirse, değiştirmeden önce gerekçesiyle raporla.
> **Bağımlılıklar:** M1 + M2 (C# tarafı). Python tarafı `mock_unity` ile M1'den bağımsız başlayabilir.
> **Teslim edilen (sonraki aşamaya):** M5: `Builds/RaceEnv/RaceEnv.exe`, `racing_rl.bridge` (`UnityVecEnv`, `MultiUnityVecEnv`, `OnnxPolicy`).
> **Kural:** DoD'daki tüm maddeler geçmeden aşama tamamlanmış sayılmaz.

**Topoloji:**
- Python **sunucudur**. `127.0.0.1:port` adresini bind eder, ardından Unity'yi başlatır.
- Unity **istemcidir**:
  - Argümanlar: `-bridgePort <p>` (varsayılan 6005), `-numAgents <N>`, `-batchmode -nographics -logFile <yol>`
  - Editor'de Play'e basıldığında 6005 portuna bağlanır.
- Her iki tarafta `TCP_NODELAY` açık (Nagle kapalı, aksi halde 40 ms gecikme olur). Tampon boyutu 1 MiB.
- Sıkı istek-yanıt: pipelining yok. Paralellik ayrı süreçler arasında sağlanır.

**Çerçeve başlığı (16 B, little-endian; Python `struct.Struct("<IHHII")`):**

| off | tip | alan |
|---|---|---|
| 0 | u32 | magic = `0x47414352` (bayt sırası "RCAG") |
| 4 | u16 | version = 1 |
| 6 | u16 | msg_type |
| 8 | u32 | seq |
| 12 | u32 | payload_len |

**Mesajlar:**

| Tip | Yön | Payload |
|---|---|---|
| `0x0001` HELLO | U→P | JSON: `{protocol, env, unity, build_id, num_agents, obs_dim:26, act_dim:2, obs_layout_hash, env_config_hash, fixed_dt, decision_period, max_episode_decisions, info_struct:"RACE_INFO_V1"}`. M6 sona `track_id, track_index, track_length_m, track_checkpoints, track_half_width, track_hash` alanlarını ekler (C0.20). |
| `0x0002` CONFIG | P→U | JSON: `{expected_obs_layout_hash, expected_env_config_hash, strict:true}`. M6: isteğe bağlı `expected_track_id`. |
| `0x0003` READY | U→P | JSON: `{ok:true}` veya ERROR |
| `0x0010` RESET | P→U | `<qII`: seed i64, start_mode u32, max_laps u32 (0 = sınırsız). Bir sonraki RESET'e kadar kalıcıdır; oto-resetler de bu değerleri kullanır. |
| `0x0011` STEP | P→U | `f32[N·2]` |
| `0x0020` STATE | U→P | RESET ve STEP yanıtı (aşağıda) |
| `0x0030` CLOSE | P→U | boş. Unity `Application.Quit(0)` çağırır. |
| `0x00FF` ERROR | ↔ | JSON: `{code, message, fatal}`. Kodlar: `BAD_MAGIC`, `BAD_VERSION`, `HASH_MISMATCH`, `BAD_ACTION`, `BAD_LENGTH`, `UNEXPECTED_MSG`; M6: `UNKNOWN_TRACK` (HELLO yerine), `TRACK_MISMATCH` |

**STATE payload (N ajan, O = 26):**

```
obs        f32[N,O]
reward     f32[N]         // K alt adımın toplamı
terminated u8[N]
truncated  u8[N]
pad        (4 − 2N mod 4) mod 4 bayt
final_obs  f32[N,O]       // yalnızca term|trunc olanlarda geçerli, diğerleri 0
info       RACE_INFO_V1[N]  (40 B, '<BBHHHifffffff'):
  0 u8 term_reason | 1 u8 lap_completed | 2 u16 laps | 4 u16 next_cp | 6 u16 reserved
  8 i32 ep_decisions | 12 f32 last_lap_s(NaN) | 16 f32 best_lap_s(NaN) | 20 f32 speed_mps
  24 f32 progress(s/L) | 28 f32 ep_return(C# kümülatif) | 32 f32 pos_x | 36 f32 pos_z
```

- Done olan ajanın `info` alanı **biten episode'a** aittir (`final_info`).
- N = 16 için payload **4064 B**'dir (1664 obs + 64 reward + 32 bayrak + 0 pad + 1664 final_obs + 640 info). İlk sürümdeki "4448" hesap hatasıydı (M3, C0.17).
- Python tarafı `np.frombuffer` ile sıfır kopyalı görünüm alır.

**Lockstep kuralları:**
- Python `seq` sayacı 1'den başlar ve istek başına 1 artar. Unity aynı değeri yankılar. Farklılık `DesyncError` ile sonuçlanır (ölümcül).
- Unity, STEP almadan fizik adımlamaz.
- Bridge `Update()` döngüsü: batchmode'da her karede bir istek, Editor'de 50 ms'lik bütçe içinde birden çok istek işlenir. Sıra: bloklu okuma, ardından RESET → `ResetAll` + STATE; STEP → C0.8'deki M3 sırası; CLOSE → çıkış.
- Tohumlama: ajan i için `DeterministicRng(seed + i)`. Spawn'lar tamamen tohuma bağlıdır.

**C# dosyaları:**
- `Bridge/{BridgeProtocol.cs (sabitler ve BinaryPrimitives okuma/yazma), BridgeClient.cs (TcpClient, NoDelay, ReadExactly, ReadTimeout 120 s, bağlantıda 30×1 s yeniden deneme), BridgeDriver.cs, CommandLineArgs.cs}`
- `Editor/BuildScript.cs`: `-executeMethod Racing.Editor.BuildScript.BuildBridge` ile `Builds/RaceEnv/RaceEnv.exe` üretir.
- Önceden ayrılmış `byte[]` tamponlar ve `Buffer.BlockCopy` kullanılır.

**Python dosyaları (`racing_rl/bridge/`):**

```python
protocol.py      # HEADER, MSG_*, RACE_INFO_DTYPE (np.dtype, itemsize==40), OBS_LAYOUT_V1, layout_hash()
transport.py     # FramedSocket: send(msg_type, payload), recv() -> (type, seq, memoryview); recv_exactly (recv_into)
unity_process.py # UnityProcess(exe, port, num_agents, log_path): Popen, bekleme, log kuyruğu, kill_tree (taskkill /T /F)
errors.py        # UnityLaunchError, UnityTimeoutError, UnityCrashedError, DesyncError, ProtocolMismatchError
vec_env.py       # UnityVecEnv(gymnasium.vector.VectorEnv)
multi_env.py     # MultiUnityVecEnv: K süreç; step_async hepsine gönderir, step_wait hepsinden toplar
gym_env.py       # RacingEnv(gymnasium.Env): N=1 sarmalayıcı
mock_unity.py    # Unity'siz test için protokolü konuşan Python istemci (dairesel pist + nokta-kütle kinematiği)
policies.py      # Policy protokolü; OnnxPolicy (onnxruntime, ML-Agents ONNX)
```

```python
class UnityVecEnv(gymnasium.vector.VectorEnv):
    metadata = {"autoreset_mode": gymnasium.vector.AutoresetMode.SAME_STEP}
    # single_observation_space = Box(low=OBS_LOW, high=OBS_HIGH, shape=(26,), float32)  # C0.6 sınırları
    # single_action_space      = Box(-1, 1, (2,), float32)
    def __init__(self, exe_path: str | None, num_agents=16, port=6005, expected_env_hash: str | None = None,
                 log_dir="runs/unity_logs", step_timeout_s=60.0, launch_timeout_s=180.0): ...
    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:  # options: {"start_mode":0|1,"max_laps":int}
    def step(self, actions) -> tuple[obs, rew, terminated, truncated, infos]
        # infos: tüm RACE_INFO alanları dizi olarak + "final_obs"[N,O] + "final_info" + "_final_obs" maskesi
    def step_async(self, actions); def step_wait(self); def close(self)
```

- **Gymnasium 5'li dönüşü:** `step()` şunu döndürür: `obs, reward, terminated, truncated, info`. Brief'teki 4'lü `obs, reward, done, info` biçimi `done = terminated or truncated` ile elde edilir. Ancak GAE için ikisinin ayrımı **korunmalıdır**.

**Hata yönetim matrisi:**

| Durum | Tespit | Aksiyon |
|---|---|---|
| Unity açılmadı | Popen hatası veya 180 s accept zaman aşımı | `UnityLaunchError` (Unity log'unun son 50 satırıyla). Farklı portla 1 kez yeniden dene. |
| Port dolu | bind sırasında `EADDRINUSE` | port+1..+20 aralığını tara |
| Handshake uyumsuz | protocol, obs_dim veya hash farklı | CLOSE gönder, süreci öldür, `ProtocolMismatchError` |
| Step zaman aşımı | 60 s (Editor'de 600 s) | `UnityTimeoutError`, süreç öldürülür. Üst katman yeniden başlatır ve reset eder. |
| Bağlantı koptu | 0 bayt veya `ConnectionResetError` | `UnityCrashedError`, yeniden başlatma (en fazla 3 kez/saat) |
| seq uyumsuz | başlık kontrolü | `DesyncError` (ölümcül) |
| NaN aksiyon | Python'da `np.isfinite` doğrulaması; Unity'de `ERROR(BAD_ACTION)` | istisna |
| NaN gözlem | Unity PhysicsError ile reset eder; Python `np.isfinite` sayar | uyarı ve sayaç |
| Zombie süreç | `atexit`, `__del__`, `finally` bloğu | CLOSE → 5 s bekle → `kill_tree` |
| Unity tarafında soket kapanması | `IOException` | Build'de `Application.Quit(2)`; Editor'de Play modundan çıkılır |

**Testler ve DoD:**

1. `pytest -m "not unity"`: protokol kodlama/çözme, `struct.calcsize("<IHHII") == 16`, `RACE_INFO_DTYPE.itemsize == 40`. `mock_unity` ile reset/step/auto-reset/final_obs/desync/timeout senaryoları.
2. `gymnasium.utils.env_checker.check_env(RacingEnv(exe, num_agents=1))` geçer.
3. **Kayıpsızlık:** N = 16, 100k karar. Desync ve zaman aşımı yok. Gönderilen mesaj sayısı alınan yanıt sayısına eşit.
4. **Determinizm:** İki taze süreç, aynı tohum ve aynı `np.random.default_rng(7)` aksiyon dizisiyle 5000 karar (birkaç oto-reset içerir). obs, reward ve bayraklar `np.array_equal` ile bit düzeyinde eşit olmalı.
5. **Ödül tutarlılığı:** Python'da toplanan episode ödülü ile `info.ep_return` arasındaki fark ≤ 1e-4.
6. **Parite:** M2'nin ONNX modeli, köprü üzerinden `OnnxPolicy` ile C0.10 protokolüne göre değerlendirilir. `completion_rate` ±5 puan, flying lap medyanı ±%1 içinde M2 in-Unity sonucuyla eşleşmelidir.
   - ONNX giriş ve çıkış adları `onnx.load` ile doğrulanır. Beklenen adlar: `obs_0` → `deterministic_continuous_actions`.
7. **Performans** (tek süreç, N = 16, rastgele aksiyon, batchmode):
   - ≥ 2000 ajan-kararı/s
   - Köprü ek yükü (RTT eksi Unity simülasyon süresi): p50 < 0.3 ms, p99 < 2 ms
