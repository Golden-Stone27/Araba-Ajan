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
| Özel motor Python | 3.13 (kurulu), `.venv`. Paketler: `torch` (CPU), `gymnasium>=1.1`, `numpy>=2`, `pyyaml`, `tensorboard`, `onnxruntime`, `matplotlib`, `pytest`. 3.13 için wheel'i olmayan bir paket çıkarsa ortam 3.12'ye düşürülür. |
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

## C0.14 Dondurulmuş değerler (M2 DoD sonrası doldurulur)

| Alan | Değer |
|---|---|
| `obs_layout_hash` | `b40ca79bdba1c2c2` (M1; C# `ObservationSpec.LayoutHash` = Python `sha256(...)[:16]`, EditMode testiyle kilitli) |
| `env_config_hash` | _M2 Env Freeze ile yazılır_ |
| RewardConfig | _M2 Env Freeze ile yazılır_ |
| T_ref (s) | _M2 sonrası_ |

## C0.15 M1 uygulama notları (sonraki aşamalar için bağlayıcı)

- **Prefab yok:** Araç `VehicleFactory.Create` ile, pist `TrackRuntime.Build` ile çalışma zamanında kurulur. `RaceEnvironment.Create(track, vehicle, sim, N, seed, mode)` ortamı tamamen kodla kurar (testler ve M2/M3 sahneleri bunu kullanır). Konfigürasyon asset'leri: `Assets/Racing/Config/{SimConfig, VehicleConfig, TrackDefinition_A}.asset`.
- **Arayüz genişlemeleri (C0.7'nin üst kümesi):** `ITrack` ayrıca `PointAt(s)`, `RightAt(s)`, `RadiusAt(s)` içerir. `EpisodeSignals` ayrıca `WrongWayHeading` (2 s boyunca `cos ψ1 < −0.5`) içerir. `TrackGeometry.NextGateAfter(s)` spawn sonrası beklenen kapıyı verir. `IVehicleController.TeleportTo(pose, initialSpeed)`.
- **M2'nin dolduracağı yerler:** `RaceAgentCore.AfterPhysicsStep()` M1'de `Reward = 0` döndürür ve sonlandırma yapmaz. M2 burada `RewardCalculator` + `TerminationPolicy` çağırır, `AddReward(r)` ve `SetTermination(reason)` ile telemetriyi günceller. `RaceEnvironment.ComputeEnvConfigHash(extra)` çağrısına M2 `RewardConfig`'i ekler.
- **Spawn:** Eğitimde s, rastgele bir kapının 1 m ilerisinden bir sonraki kapının 1 m gerisine kadar seçilir (kapı düzlemine < 1 m spawn yok). Tohum sırası: kapı, konum, yanal, yön, v₀.
- **Tur sayımı:** Start çizgisi (kapı 0) bir turun tüm kapıları sırayla geçildikten sonra geçilirse tur sayılır. Pist ortasından başlayan episode'da ilk start geçişi yalnızca zamanlamayı başlatır.
- **Fizik ayarı adı:** Enhanced Determinism, `DynamicsManager.asset` içinde `m_EnableEnhancedDeterminism` alanıdır (`Racing/Setup Project (M1)` menüsü ayarlar).
- **Testleri çalıştırma (Editor açıkken):** `Racing.Editor.TestRunReporter.Run("EditMode")` veya `Run("PlayMode")`. Sonuçlar `TestResults/<Mode>_summary.txt` dosyasına yazılır.
- **M1 ölçümleri (referans):** 0→100 km/h 6.04 s. 100→0 fren mesafesi 37.7 m. PurePursuit tur süreleri 54.58 s (kalkış) / 51.78 s / 51.78 s, en yüksek hız 144 km/h. Fuzz hızı 25k ajan-fizik-adımı/s (Editor, 16 ajan; ≈5k karar/s).
