# Ortak Sözleşmeler (tek doğruluk kaynağı)

> Tüm aşamalar (M1–M5) bu dosyaya bağlıdır. Buradaki bir değeri değiştirmek, etkilenen tüm aşamaların yeniden doğrulanmasını gerektirir.
> Sayılar ve arayüzler bağlayıcıdır. Aşama dokümanları yalnızca bu dosyaya referans verir.

## C0.1 Sürüm kilitleri

| Bileşen | Sürüm / not |
|---|---|
| Unity | 6000.4.6f1, PhysX, eski Input Manager (değiştirilmez) |
| ML-Agents (Unity) | `com.unity.ml-agents` 4.0.x (Package Manager → Add by name). Mevcut `com.unity.ai.inference 2.6.1` ile çözümleme çakışırsa çözümlenen sürüm not edilir. |
| Test | `com.unity.test-framework` manifest'e eklenir (M1). |
| Baseline Python | **3.10.11** (`winget install Python.Python.3.10`), `.venv-mla`, `mlagents==1.1.0`, `torch~=2.2.1` CPU (`--index-url https://download.pytorch.org/whl/cpu`), `numpy==1.23.5`, `protobuf<3.21` |
| Özel motor Python | **3.10.11** (M3'te kullanıcı kararıyla 3.13 yerine; M4 PyTorch uyumluluğu), `.venv` (`D:\RaceAgent\.venv`). Paketler: `torch` (CPU; M4'te 2.14.0+cpu), `gymnasium>=1.1`, `numpy>=2`, `pyyaml`, `tensorboard`, `onnxruntime`, `matplotlib`, `pytest`. |
| Donanım | Yalnız CPU. Paralellik iki katmanlı: sahne içinde N ajan, K ayrı Unity süreci. |

## C0.2 Klasör düzeni (repo kökü = Unity proje kökü)

```
Assets/Racing/
  Scripts/Core/       Racing.Core.asmdef       (ML-Agents referansı YASAK)       M1, M2
  Scripts/MLAgents/   Racing.MLAgents.asmdef   refs: Racing.Core, Unity.ML-Agents M2
  Scripts/Bridge/     Racing.Bridge.asmdef     refs: Racing.Core                  M3
  Scripts/Editor/     Racing.Editor.asmdef     (TrackBuilder, BuildScript)        M1
  Tests/EditMode/  Tests/PlayMode/             Racing.Tests.*.asmdef
  Config/   SimConfig.asset VehicleConfig.asset RewardConfig.asset TrackDefinition_A.asset
  Prefabs/  RaceCar.prefab  RaceCar_MLA.prefab (variant)  Track_A.prefab
  Scenes/   Sandbox.unity (M1)  Race_MLAgents.unity (M2)  Race_Bridge.unity (M3)
python/  pyproject.toml  racing_rl/{bridge,rl,train}/  tests/  configs/
mlagents/  config/race_ppo.yaml  requirements.txt
benchmarks/  baseline_mlagents.json  custom_ppo.json  models/  REPORT.md
docs/  README.md  contracts.md  milestones/M1..M5.md
Builds/  (gitignore)
```

## C0.3 SimConfig: tüm aşamalar için sabit

| Parametre | Değer |
|---|---|
| `fixedDeltaTime` | 0.02 s (50 Hz) |
| Karar periyodu K | 5 fizik adımı: Δt_dec = 0.1 s (10 Hz). Karar araları aynı aksiyonla doldurulur (action repeat). |
| `MaxEpisodeDecisions` | 3000 (300 s simülasyon). Aşıldığında **truncated** olur. |
| `NoProgressTimeout` | Yeni checkpoint geçilmeden 8 s simülasyon geçerse STUCK olur. |
| `NumAgentsPerEnv` N | 16. `-numAgents` argümanıyla değişir. Araç-araç çarpışması kapalı. |
| WheelCollider | `ConfigureVehicleSubsteps(5f, 12, 15)` |
| Physics | `simulationMode = Script` (**M1, M2 ve M3'ün hepsinde**), Enhanced Determinism **açık**, `autoSyncTransforms = false` |
| Player | `runInBackground = true`, `targetFrameRate = -1`, `vSyncCount = 0` |
| Zaman ölçeği | M2'de `time_scale = 20` (FixedUpdate sıklaşır). M3 ve M5'te zaman ölçeği anlamsızdır: script modunda simülasyon sınırsız hızda adımlanır. |
| Simülasyon saati | `stepIndex × fixedDeltaTime`. Core içinde `Time.time`, `Time.deltaTime` ve `UnityEngine.Random` **yasaktır**. |
| RNG | Core içinde `DeterministicRng` (PCG32). Ajan başına tohum = `seed + agentIndex`. |

## C0.4 Katmanlar ve fizik

- Katmanlar: `Car = 8`, `Wall = 9`, `Road = 10`. Çarpışma matrisinde Car×Car **kapalı**.
- Raycast maskesi yalnızca `Wall`. Her sorguda `QueryTriggerInteraction.Ignore` verilir.
- **Trigger collider hiç kullanılmaz.** Checkpoint geçişi geometrik olarak hesaplanır (C0.7).
- Rigidbody API'si Unity 6 adlarıyla kullanılır: `linearVelocity`, `angularVelocity`, `linearDamping`, `angularDamping`.
- Eğitimde `interpolation = None` olur. `Interpolate` yalnızca Sandbox sahnesinde insan sürüşü için açılır.
- `collisionDetectionMode = ContinuousDynamic`.

## C0.5 Aksiyon sözleşmesi

`a ∈ [-1,1]²`, float32. `a[0]` direksiyondur (+ sağ). `a[1]` gaz (+) veya frendir (−). NaN veya Inf aksiyon köprüde ERROR üretir. Heuristic modunda bu değer 0 kabul edilir. Sonra aksiyon `[-1, 1]` aralığına kırpılır.

Her fizik adımında uygulanan fonksiyon `VehicleController.ApplyAction`:

```
δ_max(v)   = lerp(30°, 8°, clamp01(v_fwd / 40))
δ_hedef    = a0 · δ_max(v)
δ         ← δ + clamp(δ_hedef − δ, −ω_δ·dt, +ω_δ·dt),   ω_δ = 200°/s  (yalnızca ön tekerler)
a1 ≥ 0 :  motorTorque_w = a1 · 600 Nm  (AWD, 4 teker × 600 = 2400 Nm), brakeTorque = 0
          v_fwd ≥ V_cap = 45 m/s ise motorTorque = 0   (hız sınırlayıcı)
a1 < 0 :  motorTorque = 0,  brakeTorque_w = |a1| · 3000 Nm.   GERİ VİTES YOK.
Aero   :  F_drag = −1.0 · |v| · v ;  F_down = −2.5 · v_fwd² · up   (adım başına rb.AddForce)
```

## C0.6 Gözlem sözleşmesi (OBS_DIM = 26, float32)

| idx | Ad | Formül | Aralık |
|---|---|---|---|
| 0–14 | `ray_i` | θ_i = −90° + i·(180°/14). Başlangıç noktası: `pos + up·0.5 + fwd·2.1`. Yön yalnızca yaw'a bağlıdır: `Quaternion.Euler(0, yaw+θ_i, 0)·forward`. Değer: isabet varsa `dist/50`, yoksa `1.0`. | [0, 1] |
| 15 | `v_fwd` | `clip(v_local.z / 50, −1, 1)` | [−1, 1] |
| 16 | `v_lat` | `clip(v_local.x / 50, −1, 1)` | [−1, 1] |
| 17 | `yaw_rate` | `clip(ω_local.y / 3, −1, 1)` | [−1, 1] |
| 18, 19 | `sin ψ1`, `cos ψ1` | ψ1 = XZ düzleminde araç ileri yönü ile orta çizgi teğeti `T(s_car)` arasındaki işaretli açı | [−1, 1] |
| 20, 21 | `sin ψ2`, `cos ψ2` | ψ2 = aynı açı, ileri bakış noktası `T(s_car + 30 m)` için | [−1, 1] |
| 22 | `e_lat` | `clip(yanal_sapma / 6, −1.5, 1.5)`. İşaretli, + sağ taraf. | [−1.5, 1.5] |
| 23, 24 | `prev_steer`, `prev_thr` | Son uygulanan (kırpılmış) aksiyon | [−1, 1] |
| 25 | `grounded` | Yere temas eden teker sayısı / 4 | [0, 1] |

Kanonik düzen dizesi. C# ve Python bu dizeyi birebir kullanır:

`RACE_OBS_V1|n=26|rays=15,fov=180,max=50,h=0.5|vfwd/50|vlat/50|yaw/3|psi1(sin,cos)@0|psi2(sin,cos)@30|elat/6|prev(steer,thr)|grounded/4`

`obs_layout_hash`, bu dizenin UTF-8 SHA-256 özetinin ilk 16 hex karakteridir.

## C0.7 Core C# arayüzleri (`namespace Racing.Core`)

```csharp
public readonly struct VehicleAction { public readonly float Steer, Throttle; public VehicleAction Sanitized(); }
public interface IVehicleInputSource { VehicleAction ReadAction(); }   // Keyboard, PurePursuit, Replay, Zero
public interface IVehicleController {
    void ApplyAction(in VehicleAction a, float dt);   // fizik ÖNCESİ (tork, direksiyon, aero)
    void TeleportTo(in Pose p);                        // hız=0, teker sıfırla, Physics.SyncTransforms()
    VehicleState ReadState();                          // fizik SONRASI
    VehicleAction LastApplied { get; }
}
public struct VehicleState { Vector3 Position; Quaternion Rotation; Vector3 LocalVelocity;
                             Vector3 LocalAngularVelocity; int GroundedWheels; float SteerDeg; bool IsFinite; }
public interface ITrack {
    float Length { get; } float HalfWidth { get; } int CheckpointCount { get; }
    Checkpoint GetCheckpoint(int i);
    TrackProjection Project(Vector3 p, int sampleHint);  // pencere araması (±20 örnek); hint yoksa tam tarama
    Vector3 TangentAt(float s);
    Pose SpawnPose(float s, float lateral, float headingOffsetDeg);
}
public readonly struct Checkpoint { int Index; Vector3 Position, Forward, Right; float HalfWidth, S; }
public readonly struct TrackProjection { float S, Lateral; Vector3 Tangent; int SampleIndex; }
public enum CheckpointEvent : byte { None, Passed, LapCompleted, WrongWay }
public sealed class CheckpointTracker {  // geometrik: segment (p_prev→p_curr) kapı düzlemini kesiyor mu?
    int NextIndex, LapsCompleted;
    CheckpointEvent Update(Vector3 pPrev, Vector3 pCurr);
    // İleri geçiş: kapı NextIndex için dot(p−c, n) önce <0 sonra ≥0 olur ve |lateral| ≤ halfWidth: Passed.
    // Kapı NextIndex−1 ters yönde geçilirse: WrongWay. Diğer kapılar yok sayılır (sıra zorunludur).
    void Reset(int startCheckpoint);
}
public sealed class LapTimer { void Tick(float simDt); void MarkLap(); float Current, Last, Best; } // sim saati
public enum TermReason : byte { None=0, Wall=1, WrongWay=2, Stuck=3, Flip=4, OutOfBounds=5,
                                TimeLimit=6, PhysicsError=7, Finished=8 }
public struct EpisodeSignals { bool WallContact, Flipped, OutOfBounds, NoProgressTimeout, NonFinite;
                               CheckpointEvent Cp; float WallClearance; } // M1 üretir, M2 tüketir
public struct AgentStepResult { float Reward; bool Terminated, Truncated; TermReason Reason;
                                bool LapCompleted; float LapTime; }
public sealed class RaceAgentCore : MonoBehaviour {   // 1 araç = 1 ajan, motordan bağımsız
    int Index { get; }
    void BeginEpisode(in SpawnSpec spawn);
    void ApplyAction(in VehicleAction a);              // fizik öncesi
    AgentStepResult AfterPhysicsStep();                // fizik sonrası: sinyaller → RewardCalculator (M2)
    void WriteObservation(float[] dst, int offset);    // 26 float, sıfır GC
    AgentTelemetry Telemetry { get; }                  // C0.9'daki info alanları
}
public sealed class RaceEnvironment : MonoBehaviour {   // N ajanı indeks sırasıyla tutar
    IReadOnlyList<RaceAgentCore> Agents;
    void ResetAll(long seed, StartMode mode, int maxLaps);
    void ResetAgent(int i);
    AgentStepResult[] PhysicsStep();   // ApplyAction'lar zaten yapılmış olur: Simulate(dt) + her ajan için AfterPhysicsStep
    string EnvConfigHash { get; }
}
public enum StartMode : uint { TrainRandom = 0, EvalGrid = 1 }
```

## C0.8 Kanonik adım sırası: tek doğruluk kaynağı

```
PhysicsStep():
  (1) her ajan için controller aero kuvvetlerini uygular (ApplyAction içinde)
  (2) Physics.Simulate(0.02)          // OnCollisionEnter/Stay bu çağrının içinde, olay tamponuna yazılır
  (3) i = 0..N−1 (SIRAYLA): r_i = core_i.AfterPhysicsStep()
  (4) stepIndex++
M2 (ML-Agents), SimulationDriver.FixedUpdate:
  Academy.Instance.EnvironmentStep()   // AutomaticSteppingEnabled=false; karar sınırında obs → policy;
                                        // her adımda OnActionReceived → core.ApplyAction
  results = env.PhysicsStep()
  her i için: agent.AddReward(r.Reward); r.Terminated ise EndEpisode(), r.Truncated ise EpisodeInterrupted()
M3 (Bridge), STEP mesajı gelince:
  k = 0..K−1 için: her i için core.ApplyAction(done_i ? Zero : a_i); results = env.PhysicsStep(); ödülü biriktir;
                  done olan ajan için: final_obs ve final_info yakalanır, ResetAgent(i) çağrılır,
                  done_i = true (ajan blok sonuna kadar sıfır aksiyonla bekler, ML-Agents davranışıyla aynı)
  STATE gönder
```

## C0.9 Sonlanma kodları

| Kod | Tespit (M1 sinyali) | Tür | Ceza (M2) |
|---|---|---|---|
| 1 Wall | Gövde collider'ı `Wall` katmanına `OnCollisionEnter` veya `OnCollisionStay` ile değer | terminated | −1.0 |
| 2 WrongWay | Tracker WrongWay döndürür **veya** 2 s boyunca `cos ψ1 < −0.5` | terminated | −1.0 |
| 3 Stuck | 8 s simülasyon boyunca yeni checkpoint yok | terminated | −0.5 |
| 4 Flip | 1 s boyunca `dot(up, Y) < 0.3` | terminated | −1.0 |
| 5 OutOfBounds | `y < −5` veya `|lateral| > W/2 + 3` | terminated | −1.0 |
| 6 TimeLimit | `episodeDecisions ≥ 3000` | **truncated** | 0 |
| 7 PhysicsError | NaN/Inf durum veya `|pos| > 1e4` | terminated | 0 (loglanır, oran %0.1'i geçerse M1 hatası sayılır) |
| 8 Finished | Yalnız eval modunda, `maxLaps` tamamlandığında | truncated | 0 |

## C0.10 Benchmark protokolü ve JSON şeması

- **Eval:** `StartMode.EvalGrid` ile s = 1 m'den (`SpawnSampler.EvalStartS`; start çizgisinin 1 m ilerisi, süspansiyon otururken yanlış WrongWay tetiklenmesin diye), hız 0 ile başlanır. Tur saati spawn anında başlar. Tohumlar 1000..1019 (20 episode). Ajan i'nin tohumu `1000 + i`. Pertürbasyon: yanal ~ U(−0.5, 0.5) m, yön ~ U(−2°, 2°). `maxLaps = 3`. Politika **deterministiktir**: ortalama aksiyon (μ) kullanılır. Her ajanın yalnızca ilk episode'u sayılır.
- **Metrikler:**
  - `completion_rate`: 3 turu sonlanmadan bitiren episode oranı
  - `flying_lap_median_s`: 2. ve 3. turların medyanı
  - `flying_lap_best_s`
  - `lap1_median_s`
  - `mean_speed_mps`
  - `term_reasons` histogramı
  - `steer_smoothness`: ortalama |Δa0|
  - `sector_times`: L/3 ve 2L/3 noktalarına göre
- **T_ref tanımı:** Her eğitim tohumu için başarılı episode'ların flying lap medyanı alınır. T_ref bu değerlerin tohumlar üzerinden medyanıdır.

```json
{ "schema":"race-benchmark/v1", "policy":"mlagents-ppo|custom-ppo", "evaluator":"unity-inproc|bridge",
  "env_config_hash":"…", "obs_layout_hash":"…", "unity":"6000.4.6f1", "build_id":"…",
  "train_seeds":[1,2,3],
  "eval":{"episodes":20,"seed_base":1000,"laps":3,"start":"grid","deterministic":true},
  "per_seed":[{"seed":1,"completion_rate":0.95,"flying_lap_median_s":0.0,"flying_lap_best_s":0.0,
               "lap1_median_s":0.0,"mean_speed_mps":0.0,"term_reasons":{"wall":1},"sector_times_s":[0,0,0]}],
  "summary":{"T_ref_s":0.0,"completion_rate_median":0.0},
  "training":{"total_env_decisions":0,"decisions_to_first_3lap":0,"decisions_to_95pct":0,"wallclock_h":0.0} }
```

## C0.11 Env Freeze ve config hash

- `env_config_hash` şöyle hesaplanır: SimConfig, VehicleConfig, RewardConfig, TrackDefinition (kimlik ve kontrol noktaları) ve `obs_layout_hash`, anahtarları sıralı kanonik JSON'a çevrilir. Float'lar `ToString("R", InvariantCulture)` ile yazılır. Sonuç SHA-256'nın ilk 16 hex karakteridir.
- **M2 DoD'dan sonra bu hash dondurulur.** Değişirse M2 baseline'ı yeniden koşturulur. `compare.py`, hash'i farklı iki raporu kıyaslamayı reddeder.
- Köprü, HELLO mesajında hash'i gönderir. Python beklenen hash'i konfigürasyondan okur, uyuşmazlıkta hata verir.

## C0.12 Bağımlılık grafiği ve paralellik

```
M1 ──► M2 ──► M3 (C#) ──► M5
             ▲           ▲
M3 (Python, mock ile) ───┤
M4 (yalnız contracts.md'ye bağlı, 1. günden paralel) ─┘
```

Aşamalar arası teslimler:
- M1 → M2: Core ve sahneler
- M2 → M3: dondurulmuş RewardConfig, hash, ML-Agents ONNX
- M3 → M5: build ve `UnityVecEnv`
- M4 → M5: `racing_rl.rl`

## C0.13 Genel kurallar ve riskler

- **Kültür (tr-TR):**
  - Başlangıçta `CultureInfo.DefaultThreadCurrentCulture = InvariantCulture` ayarlanır.
  - Tüm float formatlama ve ayrıştırma işlemleri `InvariantCulture` ile yapılır.
  - `ToLower` ve `ToUpper` yerine `ToLowerInvariant` kullanılır. String karşılaştırmaları `Ordinal` olur.
  - CSV ve JSON dosyalarında ondalık ayırıcı nokta olmalıdır.
- **Yol:** Proje yolu boşluk ve ASCII dışı karakterler (ş, ı) içeriyor. Tüm argümanlar tırnak içinde verilir. Bir araç sorun çıkarırsa ASCII bir junction oluşturulur: `mklink /J D:\RaceAgent "D:\Unity Projects\Araba Yarışı Ajanı"`.
- **Core içi kurallar:**
  - Adım başına GC ayırması olmaz (önceden ayrılmış `float[]` kullanılır).
  - `FindObjectsOfType` sırasına güvenilmez. Ajanlar açık bir indeks listesinde tutulur.
  - Core'da çoklu iş parçacığı kullanılmaz.
- **Editörü açıkken test:** Unity MCP araçları (`Unity_RunCommand`, `Unity_GetConsoleLogs`, `Unity_SceneView_Capture*`) kullanılabilir. Batchmode ile `-runTests` çalıştırmak için Editor'ün kapalı olması gerekir (proje kilidi).

## C0.14 Dondurulmuş değerler (Env Freeze: 2026-09-23, M2)

> **Env Freeze yürürlükte.** Sim/Vehicle/Reward/Track asset'leri, gözlem düzeni veya Core adım/ödül/sonlanma mantığı değişirse hash değişir, EditMode testi `EnvConfigHash_MatchesFrozenValue` kırılır ve M2 baseline'ı yeniden koşturulmalıdır.

| Alan | Değer |
|---|---|
| `obs_layout_hash` | `b40ca79bdba1c2c2` (M1; C# `ObservationSpec.LayoutHash` = Python `sha256(...)[:16]`, EditMode testiyle kilitli) |
| `env_config_hash` | `90240ee2b1a58b5b` (Sim + Vehicle + Reward + Track_A + obs layout; `RaceEnvironment.ComputeEnvConfigHash`, EditMode testiyle kilitli) |
| RewardConfig | `wSpeed=0.1, speedNorm=50, wWall=0.05, wallClearanceRef=1.5, wSmooth=0.02, wCheckpoint=0.05, wLap=2, crashPenalty=-1, stuckPenalty=-0.5` (`Assets/Racing/Config/RewardConfig.asset`) |
| T_ref (s) | **41.22** = medyan(41.28 s1@1e7, 41.22 s2@1M, 41.08 s3@1M); completion_rate 1.00 / 1.00 / 1.00 (`benchmarks/baseline_mlagents.json`) |

## C0.15 M1 uygulama notları (sonraki aşamalar için bağlayıcı)

- **Prefab yok:** Araç `VehicleFactory.Create` ile, pist `TrackRuntime.Build` ile çalışma zamanında kurulur. `RaceEnvironment.Create(track, vehicle, sim, N, seed, mode)` ortamı tamamen kodla kurar (testler ve M2/M3 sahneleri bunu kullanır). Konfigürasyon asset'leri: `Assets/Racing/Config/{SimConfig, VehicleConfig, TrackDefinition_A}.asset`.
- **Arayüz genişlemeleri (C0.7'nin üst kümesi):** `ITrack` ayrıca `PointAt(s)`, `RightAt(s)`, `RadiusAt(s)` içerir. `EpisodeSignals` ayrıca `WrongWayHeading` (2 s boyunca `cos ψ1 < −0.5`) içerir. `TrackGeometry.NextGateAfter(s)` spawn sonrası beklenen kapıyı verir. `IVehicleController.TeleportTo(pose, initialSpeed)`.
- **M2'nin dolduracağı yerler:** `RaceAgentCore.AfterPhysicsStep()` M1'de `Reward = 0` döndürür ve sonlandırma yapmaz. M2 burada `RewardCalculator` + `TerminationPolicy` çağırır, `AddReward(r)` ve `SetTermination(reason)` ile telemetriyi günceller. `RaceEnvironment.ComputeEnvConfigHash(extra)` çağrısına M2 `RewardConfig`'i ekler.
- **Spawn:** Eğitimde s, rastgele bir kapının 1 m ilerisinden bir sonraki kapının 1 m gerisine kadar seçilir (kapı düzlemine < 1 m spawn yok). Tohum sırası: kapı, konum, yanal, yön, v₀.
- **Tur sayımı:** Start çizgisi (kapı 0) bir turun tüm kapıları sırayla geçildikten sonra geçilirse tur sayılır. Pist ortasından başlayan episode'da ilk start geçişi yalnızca zamanlamayı başlatır.
- **Fizik ayarı adı:** Enhanced Determinism, `DynamicsManager.asset` içinde `m_EnableEnhancedDeterminism` alanıdır (`Racing/Setup Project (M1)` menüsü ayarlar).
- **Testleri çalıştırma (Editor açıkken):** `Racing.Editor.TestRunReporter.Run("EditMode")` veya `Run("PlayMode")`. Sonuçlar `TestResults/<Mode>_summary.txt` dosyasına yazılır.
- **M1 ölçümleri (referans):** 0→100 km/h 6.04 s. 100→0 fren mesafesi 37.7 m. PurePursuit tur süreleri 54.58 s (kalkış) / 51.78 s / 51.78 s, en yüksek hız 144 km/h. Fuzz hızı 25k ajan-fizik-adımı/s (Editor, 16 ajan; ≈5k karar/s).

## C0.16 M2 uygulama notları

- **Sürüm sapması (C0.1):** `com.unity.ml-agents` **4.1.0** kullanılıyor (4.0.3 ile player build `Google.Protobuf_Packed.dll` gölgelenmesi yüzünden derlenmiyor, ml-agents #6310). Communicator API 1.5.0, `mlagents==1.1.0` ile uyumlu. 4.1.0'daki resmi düzeltme Unity 6000.4.6f1'de yetmediği için `com.unity.ai.inference` 2.6.1 **gömülü paket** olarak `Packages/` altında; tek fark `Editor/ONNX/Google.Protobuf_Packed.dll.meta` (Standalone kapalı). Ayrıntı: `Packages/com.unity.ai.inference/EMBEDDED_PATCH.md`.
- **ASCII yol zorunlu (C0.13):** gRPC native DLL'i ASCII dışı yolda yüklenmiyor. `D:\RaceAgent` junction'ı kullanılır. Eğitim build'i `D:\RaceAgent\Builds\RaceEnv_MLA\RaceEnv.exe` yolundan başlatılır. Editor'e bağlanarak eğitim için Unity projesi `D:\RaceAgent` yolundan açılmalıdır.
- **Sayaçlar:** `EpisodeMonitor` zamanlayıcıları artık tamsayı fizik adımı sayar (`round(timeout/dt)`). Float birikimiyle 8 s Stuck 401. adımda (81. karar) tetikleniyordu; şimdi tam 400. adımda (80. karar).
- **Öncelik:** Aynı adımda birden çok sinyal: PhysicsError > Wall > Flip > OutOfBounds > WrongWay > Stuck > Finished > TimeLimit (`TerminationPolicy`).
- **Core API eklemeleri:** `RaceEnvironment.Create/Initialize(..., RewardConfig reward = null)`, `InitializeFromSerialized(n, seed, mode)`, `initializeOnAwake` alanı, `EnvConfigHash` (Sim+Vehicle+Reward+Track+obs). `RaceAgentCore.MaxLaps`, `EpisodeRewards`/`LastRewards` (`RewardBreakdown`). `AfterPhysicsStep` artık `Reward`, `Terminated`, `Truncated`, `Reason` doldurur; çekirdek kendini sıfırlamaz.
- **Prefab yok (C0.15 ile tutarlı):** `RaceCar_MLA.prefab` yerine `RaceAgent.Attach(core, K, model, type, deterministic)` bileşenleri koddan ekler. Sahne: `Scenes/Race_MLAgents.unity` (`Racing/Setup ML-Agents Scene (M2)`).
- **Ortam tohumu:** Player'da `-envSeed S`; verilmezse `--mlagents-port × 1000` (paralel worker'lar farklı spawn akışı alır). `-numAgents N` desteklenir.
- **`env_config_hash`:** `90240ee2b1a58b5b`, C0.14'te donduruldu.
- **Baseline eğitim protokolü:** Tüm tohumlar aynı `race_ppo.yaml` (`max_steps: 1e7`, lineer lr/β/ε) ile başlatıldı. Seed 1 1e7'ye kadar koştu; seed 2–3 ~1M checkpoint export edilince durduruldu (`max_steps` 1M yapılsaydı lineer planlar 1M'de sıfırlanır, seed 1'le kıyaslanamazdı). Paralel koşularda `--base-port` farklıdır (5005/5105/5205).

## C0.17 M3 uygulama notları

- **Python ortamı:** `D:\RaceAgent\.venv`, Python 3.10.11. Paketler: numpy 2.2.6, gymnasium 1.3.0, onnxruntime 1.23.2, onnx 1.23.0, pytest 9.1.1. Kurulum: `pip install -e python`. torch M4'te eklenir.
- **Build:** `Builds/RaceEnv/RaceEnv.exe`, sahne `Scenes/Race_Bridge.unity` (`Racing/Setup Bridge Scene (M3)`). Build komutu: `Racing.Editor.BuildScript.BuildBridge` (menü, MCP veya `-executeMethod`). TCP köprüsü ASCII dışı yolda da çalışır.
- **seq numaraları:** HELLO `seq=0` ile gider. İlk istek CONFIG'dir (`seq=1`) ve READY bu değeri yankılar. RESET/STEP `2`'den başlayarak artar.
- **STATE boyutu:** N = 16 için 4064 B.
- **`lap_completed`:** K alt adım üzerinden OR'lanır.
- **Done olan ajanın bilgileri:** `final_obs` ve `info` biten episode'a aittir.
- **Blok içi reset ve ödül aktarımı:** Sıfırlanan ajan blok sonuna kadar sıfır aksiyonla bekler. Bu sürede kazandığı ödül yeni episode'a aittir ve **bir sonraki STATE'in ödülüne eklenir**. Böylece Σ reward = `ep_return` olur. Bekleme sırasında yeni episode da biterse (beklenmez), ajan yeniden sıfırlanır ve durum `double_done` sayacına yazılır.
- **`ep_return`:** `RaceAgentCore` içinde double ile biriktirilir. float32 birikim 15k adımda ~1e-3 kayıyordu. Yalnız telemetriyi etkiler; ödüller ve hash değişmez.
- **RESET = taze araçlar (`RaceEnvironment.RebuildAgents`):**
  - **Bulgu:** `TeleportTo`, WheelCollider'ın iç PhysX durumunu (süspansiyon, temas, lastik) sıfırlamıyor. Teleport ile yapılan reset'ler bu yüzden önceki episode'a bağlı kalıyor: aynı tohumla fark ~0.5 (gözlem biriminde).
  - **Denenen çözüm:** `TeleportTo` içinde tekerleri kapatıp açmak ilk episode dinamiğini değiştirdi (s1 flying lap 41.28 → 41.12 s). Env Freeze nedeniyle geri alındı.
  - **Uygulanan çözüm:** Açık RESET mesajında köprü araçları yeniden kurar. Böylece `RESET(seed)` aynı süreç içinde bit düzeyinde tekrarlanabilir olur (`check_env` geçer) ve M2 eval'inin başlangıç koşuluyla birebir aynıdır.
  - **Kapsam:** Oto-reset'ler (STEP içi) Core teleport'unu kullanmaya devam eder; M2 eğitimiyle aynıdır. Eğitim (M5) için bu tarihçe bağımlılığı yalnızca bir gürültü kaynağıdır. İki taze süreç arasındaki determinizm bundan etkilenmez.
  - **Kural:** `RebuildAgents` ML-Agents sahnesinde kullanılmaz.
- **`-bridgeTimingLog <yol>`:** İsteğe bağlıdır.
  - Unity her istek için işleme süresini ikili dosyaya yazar: kayıt başına `'<IHxxf'` (seq, msg_type, unity_us).
  - `<yol>.json` yan dosyası şunları içerir: istek ve mesaj sayaçları, `gc_steady_bytes` (ilk 100 STEP sonrası `GC.GetAllocatedBytesForCurrentThread` farkı), `double_done`.
  - Köprü ek yükü = Python RTT − unity_us, seq bazında eşlenir.
- **Batchmode:** Karede bir istek işlenir. Editor: 50 ms bütçe. Python'da `exe_path=None` Editor'ün Play ile bağlanmasını bekler (adım zaman aşımı 600 s).
- **`RacingEnv` (N = 1) info:** NaN tur süreleri `None` döner, çünkü gymnasium `check_env` NaN'ı eşit saymaz. `UnityVecEnv` NaN'ı korur.
- **Köprü değerlendiricisi (`racing_rl.bridge.evaluate`):** Tur süreleri kesindir (Unity sim saati). `mean_speed_mps` ve `sector_times_s` karar (0.1 s) örneklemesiyle hesaplanır; in-proc değerlendirici ise fizik adımı (0.02 s) örneklemesi kullanır.

## C0.18 M4 uygulama notları

- **Python ortamı:** `.venv` içine `torch 2.14.0+cpu` (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), `tensorboard 2.21.0`, `matplotlib 3.10.9` eklendi. numpy 2.2.6 olarak kaldı. `racing_rl` 0.4.0: `torch` zorunlu bağımlılık, `tensorboard`/`matplotlib` `[train]` ekstrası. `pytest` varsayılanı `-m "not slow"`; `testpaths` artık `racing_rl/rl/tests` dizinini de içerir.
- **Policy protokolü köprüsü:** M3 köprüsündeki `Policy` protokolü `__call__(obs)`, M4'teki `act(obs, deterministic)` biçimindedir. `TorchPolicy` ikisini de uygular; `__call__` kurucudaki `deterministic` bayrağını kullanır (varsayılan `True`, C0.10). Böylece `racing_rl.bridge.evaluate.run_eval` değişmeden çalışır. `TorchPolicy` hiçbir zaman `obs_rms`'i güncellemez.
- **`ActorCritic` eklemeleri:** `act_full(obs, update_rms, deterministic=False)` beşli döndürür: `(a_env, u, logp, v, obs_n)`. `obs_n`, buffer'a yazılan toplama anı normalize gözlemidir. `act()` sözleşmedeki dörtlüyü döndürür ve `obs_n`'i `last_obs_n` alanında tutar. `act_deterministic(obs)` = transform(μ). Sonlu olmayan gözlemler RMS'e girmeden 0 ile değiştirilir ve uyarı verilir. Ortogonal başlatma tek iş parçacığında yapılır: LAPACK QR sonucu iş parçacığı sayısına bağlıdır (~1e-6), aynı tohum bu sayede her durumda bit düzeyinde aynı ağırlıkları verir.
- **Varsayılan (`default`) hiperparametreler:** `lr` 3e-4'ten lineer olarak 0'a, Adam eps 1e-5, ε 0.2 sabit, ε_v 0.2 (`clip_vloss=True`), c_e 1e-3 sabit, c_v 0.5, γ 0.99, λ 0.95, 4 epoch, minibatch 1024, T 256, `obs_clip` 10, `max_grad_norm` 0.5, `target_kl` yok, torch iş parçacığı sayısı toplamada 2, güncellemede 4. `parity` ön ayarı M4 dokümanındaki gibidir; T = `20480 // N_total` (`PPOConfig.rollout_steps(N)`).
- **DoD-7 test konfigürasyonu:** `default` ön ayarı üzerine lr 1e-3, N = 64, T = 64, minibatch 256. Ölçüt, **deterministik** politikanın (μ) taze örneklerdeki ortalama ödülüdür (8192 örnek). Ölçülen: eşik 15. güncellemede geçildi, 30. güncellemede −0.020, test 11.5 s. Stokastik ödül raporlanır ama doğrulanmaz: σ ≈ 0.3'te gürültünün beklenen cezası tek başına ≈ 2σ² ≈ 0.17'dir, bu yüzden −0.05 eşiğini stokastik politika aşamaz.
- **NaN koruması:** "Son checkpoint yüklenir" kuralı bellek içi anlık görüntüyle uygulanır. Her `update()` başında model, optimizer ve `obs_rms` kopyalanır. Kayıp veya gradyan normu sonlu değilse adım atlanır, anlık görüntü geri yüklenir, `lr_mult` yarıya iner (kalıcıdır, planın üstüne çarpılır), `RuntimeWarning` verilir ve güncelleme `nan_skipped = 1` ile sonlanır.
- **Checkpoint:** Sözleşmedeki alanlara ek olarak `format` (`racing_rl.ppo/v1`), `obs_dim`, `act_dim`, `rng{torch, numpy}` ve `extra` (`trainer` durumu: `lr_mult`, minibatch üreteci; `update`; `best.pt` için `eval`) yazılır. İçerik yalnızca tensör ve ilkel türlerden oluşur, `torch.load(weights_only=True)` ile okunur. `obs_rms` `{mean, var}` float64 tensör, `count` float'tır. `load_checkpoint(path, expected_env_hash, expected_obs_hash)` hash uyuşmazlığında `CheckpointMismatchError` verir. Yazma atomiktir (`.tmp` + `os.replace`).
- **Kapsam eklemesi:** M4 dosya listesine `rollout.py` (`collect_rollout`, `EpisodeStats`) ve `loop.py` (`train`) eklendi. `train(envs, cfg, total_steps, ckpt_dir, eval_fn, eval_every=25, ckpt_every=10, seed, log_fn, *, ac=None, ...)` kütüphane düzeyindedir (CLI ve TensorBoard M5'e aittir). `global_step` ortam kararı sayar (güncelleme başına T·N). Her 10 güncellemede `ckpt_{step}.pt` yazılır (en fazla 10 tutulur, son güncelleme de kaydedilir). Her 25 güncellemede `eval_fn(TorchPolicy)` çağrılır. C0.10 `per_seed` sözlüğü `completion_rate` azalan, sonra `flying_lap_median_s` artan sırayla (None en kötü) değerlendirilir ve en iyisi `best.pt` olur. `collect_rollout` `step_async`/`step_wait` kullanır; yalnız bu ikisi olmayan ortamlarda (testlerdeki gymnasium `SyncVectorEnv`) `step()`'e düşer.

## C0.19 M5 uygulama notları

- **Model seçimi / test ayrımı (C0.10 sapması, kullanıcı onaylı 2026-09-23):** Eğitim içi periyodik eval (her 25 güncellemede, ayrı ve kalıcı 20 ajanlı Unity süreci) **doğrulama tohumları 2000..2019** ile yapılır; `best.pt` bununla seçilir. Resmi kıyas (ML-Agents ONNX ve özel PPO) C0.10'daki **test tohumları 1000..1019** ile, aynı build ve aynı köprü değerlendiricisiyle (`racing_rl.train.evaluate`) yapılır. Protokolün geri kalanı (EvalGrid, 3 tur, deterministik μ, ilk episode) değişmez.
- **K × N:** `sps_probe.py` (rastgele aksiyon, N ∈ {8,16,32} × K ∈ {1..4}) p99 < 50 ms şartıyla en yüksek SPS'i **N = 32, K = 4** (128 ajan) verdi: 28.9k karar/s rastgele, 23.7k politika ile, p99 5.8 ms. `parity` rollout'u T = 20480 // 128 = 160.
- **Kayıp ölçeği paritesi (config ile, M4 kodu değişmedi):** ML-Agents 1.1 kaybı `L_π + 0.5·mean(max(v₁,v₂)) − β·mean_j H_j`. `racing_rl` entropiyi aksiyon boyutları üzerinden topluyor ve değer kaybına ½ çarpanı ekliyor. `configs/ppo_parity.yaml` bu yüzden `ent_coef 2.5e-3 → 5e-6` (β/2) ve `vf_coef 1.0` kullanır: kayıp terimleri ML-Agents ile birebir aynıdır. TensorBoard `Policy/Entropy` boyut ortalamasıdır (ML-Agents ile üst üste çizilebilir), toplam `Train/EntropySum`.
- **Kalan (kod düzeyinde) farklar:** avantaj normalizasyonu minibatch bazında (ML-Agents: tüm buffer), GAE rollout boyunca (T = 160; ML-Agents `time_horizon` 128), gradyan normu 0.5 ile kırpılıyor (ML-Agents kırpmıyor), değer kırpma ε_v 0.2 sabit (ML-Agents ε ile birlikte 0.2 → 0.1).
- **Eğitim telemetrisi M2 ile aynı tanımlar:** `Race/CompletionRate` = truncation ile biten episode oranı, `Race/Laps3Rate` = laps ≥ 3; `decisions_to_first_3lap` / `decisions_to_95pct` bu iki eğrinin (güncelleme başına ≈ 20k karar penceresi) ilk eşik geçişidir.
- **Çalıştırma:** `python -m racing_rl.train.train_ppo --config configs/ppo_parity.yaml --seed S --run-dir ../runs/m5/parity_sS` (`--resume` ile devam). Çıktılar `runs/` altında (gitignore); değerlendirilen `best.pt` kopyaları `benchmarks/models/custom_ppo_s{S}.pt`.
- **Sonuç ve rapor:** `benchmarks/M5_FINAL.md` (kullanıcı kararıyla `REPORT.md` yerine; önceki aşama raporlarıyla aynı adlandırma). Özel PPO `best.pt` T = **39.18 s** (39.18 / 38.68 / 40.50), completion 1.00 × 3; ML-Agents köprü üzerinden 41.28 / 41.22 / 41.08 (T_ref 41.22). Son checkpoint (10M) T = 42.66 s ve 1M anlık görüntüsü T = 43.10 s raporda ayrı sütunlarda. `racing_rl` 0.5.0.

## C0.20 M6 çoklu pist notları (devam ediyor)

- **Değişmez:** Track_A (Benchmark) M5 ile bit düzeyinde aynıdır. `TrackFingerprint.Geometry` = `f6930a41d26c9e30`, `TrackFingerprint.Physical` = `4ee36c524bd1cf90` (`TrackAGoldenTests`). `env_config_hash` `90240ee2b1a58b5b` değişmez.
- **Hash kuralı:** `TrackDefinition`'a M6'da eklenen alanlar kanonik JSON'a yalnızca varsayılan dışı değer aldıklarında yazılır. `profile` ve `notes` alanları meta veridir, hash'e girmez.
- **Katalog:** `Assets/Racing/Config/TrackCatalog.asset`. 0. indeks her zaman Track_A'dır. Yeni pistler yalnızca sona eklenir; yayımlanmış indeksler değişmez. Kimlikler ASCII ve benzersizdir (Ordinal). Menü: `Racing/Tracks/Setup Track Catalog (M6)`, `Racing/Tracks/Report Tracks (M6)`.
- **Evrensel pist kuralları:** `TrackValidator.CheckProfile`. 600 ≤ L ≤ 1600 m, R_min ≥ 12 m, öz-ayrım > W + 4, |koordinat| < 500, 10 ≤ W ≤ 14 m, toplam dönüş ±360° ± 5°. Profil kuralları `docs/milestones/M6_multitrack.md` dosyasındadır. Benchmark profili M1 ölçütleridir (`Report.Passed`). Procedural profili ayrıca sağa ve sola dönen virajlar ile R_min ≤ 40 m ister; bu, üreticinin yalnızca dışbükey ovaller üretmesini engeller.
- **`TrackLayout`:** Düzlük ve yay zinciri. Başlangıç (0, 0), yön +x, pozitif açı sola döner. Zincir tam ±360° dönmelidir. Kapanma açığı iki flex düzlükle (2×2 çözüm) kapatılır, kalan artık yay uzunluğuna dağıtılır. Kontrol noktaları sınırlayıcı kutuya göre ortalanır ve santimetreye yuvarlanır. Ölçülen R_min tasarım yarıçapının ≈ 0.93 katıdır (spline geçiş aşımı).
- **Katalog pistleri (dondurulmuş, `TrackFreezeTests`):**

| İndeks | Kimlik | Profil | W | L | Kapı | `env_config_hash` | Geometri | Fiziksel |
|---|---|---|---|---|---|---|---|---|
| 0 | Track_A | Benchmark | 12 | 1144.1 | 114 | `90240ee2b1a58b5b` | `f6930a41d26c9e30` | `4ee36c524bd1cf90` |
| 1 | Track_B | Technical | 11 | 1114.8 | 111 | `038a104393cbfb72` | `1badf7937c4d3373` | `e571cd02540992c9` |
| 2 | Track_C | Speedway | 13 | 1262.5 | 126 | `0e099647315ed638` | `73a2aed396375311` | `783476a6f82becb7` |
| 3 | Track_D | Elevation | 12 | 1094.4 | 109 | `dbce4b7f772fc7b4` | `bfa24bc57d5515dd` | `9705cfe460a64ac1` |

  - B, C ve D, `TrackLayouts.TechnicalB/SpeedwayC/HillD` düzenlerinden `Racing/Tracks/Bake Track Assets (M6)` menüsüyle bake edilir. `BakedAssets_MatchTheirLayouts` testi asset ile layout'un senkron kaldığını denetler.
  - Diğer alanlar (duvar 1.5 × 1 m, segment 2 m, kapı aralığı 10 m, örnek aralığı 1 m, α 0.5) Track_A ile aynıdır.
  - PurePursuit referans turları: B 67.08 / 65.02 / 65.00 s, C 44.14 / 40.28 / 40.26 s, D 53.96 / 51.86 / 51.86 s.
- **Usulü pistler `proc:<seed>`:**
  - Kaynak: `ProceduralTrackGenerator` v1. Tohum negatif olmayan bir ondalık int64'tür (`TryParseName`); `trackId` = `proc:<seed>`; katalog indeksi −1'dir.
  - PCG32 akışı `0x70726f63` ("proc"). Her aday için:
    - 10–18 düğüm; elips en-boy oranı 1–1.7; açı jitter'ı ±0.3 adım; radyal pertürbasyon 0.45–1.25
    - rastgele güçte düzleştirme (0–0.5)
    - rastgele rotasyon ve %50 aynalama (saat yönünde pist)
    - W 10.5–13.5 m (0.5 m adımlarla); hedef L 800–1400 m (tam ölçekleme)
    - santimetreye yuvarlama
  - Adaylar evrensel ve Procedural kurallarıyla kabul edilir veya reddedilir; ilk geçen aday döner. En fazla 500 deneme; 0..199 tohumlarında ortalama 2.9, en fazla 13 deneme.
  - Aynı tohum bit düzeyinde aynı pisti verir. Donmuş referanslar: `proc:0` `00724614a3c71752`, `proc:1` `860ce2361684922d`, `proc:7` `59966c9f08fc1027`, `proc:1000` `7b4353dd8647e3d3` (`env_config_hash`).
  - Üretici değişirse `Version` artırılır ve bu tablo güncellenir.
  - 0..99 tohumlarının yüzde 47'si saat yönündedir; PurePursuit 0..19 tohumlarının hepsinde temiz tur tamamlar.
  - **Sınırlama (v1):** Uzun düzlük yoktur; en uzun düzlük 36–149 m arasındadır. Yüksek hız genellemesi için Track_C kullanılır.
- **Kot ve eğim (Track_D, `TrackDefinition` M6 alanları):**
  - **Profil:** `elevation` alanı, `ElevationKey(u = s/L, y)` anahtarlarından oluşan periyodik bir kot profilidir. İki anahtar arasında y = y₀ + (y₁ − y₀)(1 − cos πt)/2 kullanılır; eğim anahtarlarda 0 ve C1 süreklidir. Track_D: 150 m'ye kadar düz, 480 m'de +10 m (≈%4.8), 600 m'ye kadar plato, 960 m'de 0 (≈%4.4), sonra düz.
  - **Hash:** Yeni alanlar yalnızca varsayılan dışı değer aldıklarında hash'e yazılır: `track.elevation` (anahtarların hash'i), `track.wallColliderExtraBelow/Above`, `track.roadCollider` ve `track.roadColliderMargin`. Düz pistlerin kanonik JSON'u değişmez.
  - **Geometri:**
    - s yatay (XZ) yay uzunluğudur. Örnek noktaları y = `HeightAt(s/L)` taşır.
    - Tangent 3B'dir (eğim boyunca): ödül v·t̂ yol boyunca ölçülür ve kapı düzlemi 3B tangente diktir.
    - Right vektörü, yarıçap, izdüşüm (`Project`) ve öz-ayrım yatay düzlemde hesaplanır.
    - `y ≡ 0` iken float işlemleri M5 ile bit düzeyinde aynıdır (Track_A altın testleri).
  - **Collider'lar:**
    - `RoadColliderMode.MeshStrip`: yarı genişliği W/2 + duvar kalınlığı + 3 m olan, kotu izleyen, yukarı bakan bir MeshCollider (Road katmanı). Zemin küpü 0.1 m aşağı iner.
    - Duvar collider'ları görünmez biçimde yolun 10 m altına ve üstüne uzatılır; görsel duvar 1.5 m kalır.
    - Evrensel kural: Kotlu bir pist MeshStrip kullanmalıdır. Uzatma alt tarafta ≥ eğim·50 + 1 m olmalı, üst uç ise ışın yüksekliği + eğim·50 + 1 m'yi geçmelidir. En dik eğim ≤ %6, y ≥ 0.
    - Elevation profili ayrıca kot aralığı ≥ 6 m ve en dik eğim ≥ %3 ister.
  - **Işın kanıtı (`TrackD_Rays_SeeEveryWallWithin50m_OnlyWithTheExtension`):** Referans, yatay ışının duvar iç yüzlerine (merkez çizgisi ± W/2) olan 2B kesişim mesafesidir. Karşılaştırma bir PurePursuit turu boyunca yapıldı.
    - Uzatılmış duvarlar: 50 m içinde duvar olan 7696 ışında 0 kaçırma, en büyük sapma 0.21 m.
    - M1 yüksekliğindeki duvarlar: 469 kaçırma (%6.1).
    - Gözlem sözleşmesi (C0.6) değişmez.
  - **Determinizm bulgusu (Track_D):** Aynı süreçte arka arkaya kurulan iki ortam, eğimli zeminde bit düzeyinde aynı değildir.
    - **Desen:** Sonuç, süreçte daha önce kaç kez statik pist collider'ı kurulduğuna bağlıdır ve 4 koşu periyotlu bir A A B B desenidir.
    - **Elenen nedenler:** Desen worker thread sayısından bağımsızdır (0 worker ile aynı). MeshCollider yerine BoxCollider şeridi kullanıldığında da görülür. Tekerlek temas verileri ayrışma adımında aynıdır ve gövde teması yoktur.
    - **Düz pistler etkilenmez:** Düz pistlerde (A, B, C, düz MeshStrip dahil) desen görülmez.
    - **Köprü için geçerli garantiler:** Köprü, pisti süreç başına bir kez kurar.
      - Taze süreçler aynı geçmişle başladığı için birbirinin aynısıdır.
      - Aynı ortamda `RebuildAgents` ile yapılan RESET bit düzeyinde tekrarlanabilir (`TrackD_RebuildAgents_IsBitwiseReproducible`: 3 koşu × 52 800 float aynı).
      - Süreçler arası eşitlik 4. adımda gerçek build ile doğrulandı (aşağıda, "Köprü").
- **Köprü (Adım 4):**
  - **Pist seçimi:** Player argümanları `-trackName <kimlik | asset adı | proc:<seed>>` ve `-trackIndex <i>`. Pist süreç başına bir kez kurulur; RESET pisti değiştirmez.
    - Öncelik: CLI, sonra `BridgeDriver.trackOverride` (yalnız Editor; string olduğu için `proc:` de alır), en son sahnedeki serileştirilmiş pist. Argüman yoksa M5 yolu aynen çalışır (aynı `TrackDefinition` nesnesi).
    - Ad çözümlemesi `TrackCatalog.TryResolve` ile yapılır (kimlik, asset adı, `proc:<seed>` → indeks −1). İki bayrak birlikte verilirse aynı katalog girdisini göstermelidir.
    - Katalog `BridgeDriver.catalog` alanındadır. `Race_Bridge` ve `Race_Watch` sahnelerine `Racing/Tracks/Bind Track Catalog To Bridge Scenes (M6)` menüsüyle bağlanır (sahneler yeniden üretilmez); `SetupBridgeScene` ve `SetupWatchScene` de bağlar.
  - **Hatalar:** Pist çözümlenemezse ortam kurulmaz. Unity bağlanır, HELLO yerine `ERROR{UNKNOWN_TRACK, fatal}` gönderir ve 2 koduyla çıkar. Kapsam: bilinmeyen ad (Ordinal), aralık dışı indeks, tamsayı olmayan veya değersiz bayrak, bayrak çelişkisi (ayrı kod yok, mesaj açıklar), üretilemeyen `proc:` pisti. Python bunu `RemoteError(code="UNKNOWN_TRACK")` olarak görür.
  - **HELLO:** M3 alanları aynı sıra ve biçimde kalır. Sona şunlar eklenir: `track_id`, `track_index` (proc: −1), `track_length_m` (float `R`), `track_checkpoints`, `track_half_width`, `track_hash` (= `ConfigHash.Compute(trackDef)`, yalnız pist parçası). Alanlar yalnız eklendiği için **`PROTOCOL`/`Version` 1 kalır**; M5 Python istemcisi yeni build'le değişmeden çalışır.
    - Donmuş `track_hash`: A `0f2fbf3481a1a24f`, B `6ce4f5d0b0d07eda`, C `ff0ea58517167d6f`, D `04f6036ec025016a` (`BridgeTrackTests`).
  - **CONFIG:** İsteğe bağlı `expected_track_id`. Önce pist denetlenir (başka pist hash'i de değiştirdiği için daha açıklayıcıdır): strict modda uyuşmazlık `ERROR{TRACK_MISMATCH, fatal}` verir, strict değilse uyarı yazılır. Sonra M3 hash denetimi (`HASH_MISMATCH`) gelir. JsonUtility'de eksik `strict` false okunur (M3 ile aynı).
  - **Kod:** `CommandLineArgs.TryParseTrackArgs`, `BridgeDriver.TryResolveTrack`, `BridgeProtocol.{TrackInfo, HelloJson, ConfigMsg, CheckConfig}`, `RaceEnvironment.InitializeFromSerialized(n, seed, mode, trackOverride = null)`. Python: `protocol.ERR_UNKNOWN_TRACK`, `ERR_TRACK_MISMATCH`, `HELLO_TRACK_FIELDS`; `mock_unity` yeni HELLO alanlarını üretir ve `expected_track_id`'yi aynı sırayla denetler.
  - **Doğrulama (gerçek build, `python/scripts/m6_bridge_check.py`):**
    - `smoke` (`benchmarks/eval/m6_bridge_smoke.json`): argümansız, `-trackName`/`-trackIndex` ile A–D ve `proc:7` HELLO'ları C0.20 hash'leriyle eşleşir; 5 `UNKNOWN_TRACK` ve 3 `expected_track_id` vakası beklenen yanıtı verir (16/16).
    - `regress` (`benchmarks/eval/m6_track_a_regression.json`): 6 M5 modeli (özel PPO s1–s3, ML-Agents s1–s3), argümansız ve `-trackName Track_A` ile, özel s1 ayrıca `-trackIndex 0` ile (13 koşu). Tüm per-seed metrikleri ve `benchmarks/eval/traces/*.npz` izlerinin her dizisi bit düzeyinde aynıdır: özel PPO 39.18 / 38.68 / 40.50 s, ML-Agents 41.28 / 41.22 / 41.08 s.
    - `determinism` (`benchmarks/eval/m6_track_determinism.json`): Her pist (Track_D, kontrol olarak Track_A) ve senaryo için `-trackName` ile 2 taze süreç açılır. Her süreçte 2 kez RESET(1000, TrainRandom) + 1500 STEP koşulur (16 ajan). Karşılaştırma, her STATE'in SHA-256 özetiyle yapılır (obs, ödül, bayraklar, final_obs, RACE_INFO).
      - Senaryolar: sabit tohumlu aksiyon dizisi (çok duvar teması ve oto-reset: 698 / 714 bitiş) ve M5 `custom_ppo_s1` politikası (deterministik μ; Track_D'de en yüksek 36.9 m/s, pistin 20 diliminin hepsinde 20 m/s üstü).
      - Sonuç: 1501 STATE'lik akışlar hem süreçler arasında hem süreç içindeki iki RESET arasında bit düzeyinde aynıdır. Böylece Track_D için "taze süreçler aynıdır" garantisi gerçek build ile doğrulanmıştır.
      - Not: M5 politikası Track_D'de sıfır atışta tur tamamlayamıyor (132 bitiş/koşu), Track_A'da 41 tur. Sıfır atış raporu Adım 5'e aittir.
