# M2: Doğrulama ve Ödül Mühendisliği (ML-Agents Baseline)

> **Alt ajan talimatı:** Bu aşamayı uygulamak için yalnızca bu dosya ve [`../contracts.md`](../contracts.md) yeterlidir.
> Sözleşmelerdeki (C0.x) bir değeri değiştirmen gerekirse, değiştirmeden önce gerekçesiyle raporla.
> **Bağımlılıklar:** M1 (Core, sinyaller, sahne).
> **Teslim edilen (sonraki aşamaya):** M3/M5: dondurulmuş RewardConfig ve `env_config_hash`, `baseline_mlagents.json`, ONNX modelleri.
> **Kural:** DoD'daki tüm maddeler geçmeden aşama tamamlanmış sayılmaz.

**Dosyalar:**
- `Core/Reward/{RewardConfig.cs (ScriptableObject), RewardCalculator.cs}`
- `Core/Episode/TerminationPolicy.cs`
- `MLAgents/{RaceAgent.cs, MlaSimulationDriver.cs, BenchmarkRunner.cs}`
- `Core/Benchmark/BenchmarkRecorder.cs` (JSON yazar, InvariantCulture)
- `Prefabs/RaceCar_MLA.prefab`
- `Scenes/Race_MLAgents.unity`
- `mlagents/config/race_ppo.yaml`
- `mlagents/requirements.txt`
- `benchmarks/baseline_mlagents.json`
- `benchmarks/models/mlagents_baseline_s{1,2,3}.onnx`

**Ödül (fizik adımı j başına, K = 5; karar ödülü R_t = Σ r_j):**

```
r_j =  (w_v/K) · clip( (v_j · t̂(s_j)) / 50 , −1, 1 )          // teğet yönlü (checkpoint dizisi yönünde) işaretli hız
     − (w_wall/K) · max(0, 1 − WallClearance_j / 1.5)          // duvara yakınlık (analitik)
     − w_smooth · |a0_j − a0_{j−1}|                             // yalnız karar sınırında ≠ 0
     + w_cp · 1[Passed]  +  w_lap · 1[LapCompleted]
     + R_term(reason)                                           // C0.9
```

| w_v | w_cp | w_lap | w_wall | w_smooth | R_crash (Wall, WrongWay, Flip, OOB) | R_stuck |
|---|---|---|---|---|---|---|
| 0.1 | 0.05 | 2.0 | 0.05 | 0.02 | −1.0 | −0.5 |

**Ölçek kontrolü:**
- Ortalama 28 m/s'de hız terimi karar başına yaklaşık 0.056 eder. Bir tur (~390 karar) yaklaşık 22 ödül getirir. Buna ~5.5 checkpoint ödülü ve 2 tur ödülü eklenir.
- **İntihar değişmezi:** İleri giden bir aracın karar başına ödülü > 0 olmalıdır.
- Adım başına negatif ödül üst sınırı yaklaşık 0.05 + 0.04'tür. Bu değer |R_crash| = 1'den çok küçüktür.
- Çarpmanın örtük maliyeti ≈ Σγ^t·0.056 ≈ 5.6 (γ = 0.99). Bu yüzden bilerek çarpmak asla kârlı değildir.

**Exploit önleme:**

| Exploit | Önlem |
|---|---|
| Checkpoint'te ileri-geri gidip ödül toplama | Yalnızca `NextIndex` kapısı sayılır. Ters geçiş WrongWay ile sonlanır. |
| Kısayol | Kapalı duvarlar, 10 m kapı aralığı ve sıra zorunluluğu |
| Duvara sürtünerek ilerleme | Duvar teması terminal |
| Duvara yaslanıp durarak kaçınma | Stuck sonlanması; hız ödülü 0 olur |
| Zikzak ile hız toplama | t̂ izdüşümü, direksiyon oran sınırı, smoothness cezası |
| Geri gitme | Geri vites yok; negatif teğet hız ceza verir |
| Duvar üstünden uçma | Duvar yüksekliği 1.5 m, OOB sonlanması, downforce |
| Spawn noktasını ezberleme | Eğitimde spawn: s ~ U(0, L), yanal ~ U(−1.8, 1.8) m, yön ~ U(−10°, 10°), v₀ ~ U(0, 10) m/s |

**RaceAgent : Agent (Racing.MLAgents):**
- `Initialize()`:
  - Core referansları alınır.
  - `MaxStep = 0` (truncation Core'da yapılır).
  - `BehaviorParameters`: ad "RaceCar", VectorObservationSize 26, Stacked 1, Continuous 2, `DeterministicInference = true` (eval için).
  - `DecisionRequester`: Period 5, TakeActionsBetweenDecisions = true.
- `OnEpisodeBegin()` → `core.BeginEpisode(spawnSampler.Next(i))`
- `CollectObservations(VectorSensor s)` → `core.WriteObservation(buf, 0)`, ardından `s.AddObservation(buf)`
- `OnActionReceived(ActionBuffers a)` → `core.ApplyAction(new(a.ContinuousActions[0], a.ContinuousActions[1]))`
- `Heuristic` → `KeyboardInputSource`
- `MlaSimulationDriver`:
  - `Awake`: `Academy.Instance.AutomaticSteppingEnabled = false`, `Physics.simulationMode = Script`
  - `FixedUpdate`: C0.8'deki M2 sırasını uygular
- `StatsRecorder` anahtarları: `Race/LapTime`, `Race/CompletionRate`, `Race/TermReason/<ad>`, `Race/MeanSpeed`

**`race_ppo.yaml`:**

```yaml
behaviors:
  RaceCar:
    trainer_type: ppo
    hyperparameters: {batch_size: 1024, buffer_size: 20480, learning_rate: 3.0e-4, beta: 5.0e-3,
      epsilon: 0.2, lambd: 0.95, num_epoch: 3, learning_rate_schedule: linear,
      beta_schedule: linear, epsilon_schedule: linear}
    network_settings: {normalize: true, hidden_units: 256, num_layers: 2}
    reward_signals: {extrinsic: {gamma: 0.99, strength: 1.0}}
    max_steps: 1.0e7
    time_horizon: 128
    summary_freq: 20000
    checkpoint_interval: 500000
    keep_checkpoints: 10
engine_settings: {time_scale: 20, target_frame_rate: -1, no_graphics: true}
torch_settings: {device: cpu}
```

**Çalıştırma:**
1. Editor duman testi: `mlagents-learn mlagents/config/race_ppo.yaml --run-id=smoke`, ardından Play. 200k adımda ödülün arttığı görülür.
2. Build: `Builds/RaceEnv_MLA/RaceEnv.exe`
3. Eğitim: `--env=... --num-envs=3 --no-graphics --seed={1,2,3} --run-id=baseline_s{n}`

**Öğrenilebilirlik teşhis tablosu:**

| Belirti | Olası neden | Düzeltme |
|---|---|---|
| Episode uzunluğu yalnızca birkaç karar | Spawn anında temas veya süspansiyon zıplaması | Spawn yüksekliğini ve yanal kırpmayı düzelt |
| Sürekli Stuck | w_v düşük veya w_wall baskın | w_v ↑, w_wall ↓ |
| Zikzak | Smoothness cezası yetersiz | w_smooth ↑ |
| Entropi veya σ büyüyor | Ödül çok küçük veya gürültülü | Ölçeği kontrol et, ödülü logla |
| Tur atıyor ama yavaş | Crash cezası aşırı | R_crash'i −0.5'e çek |
| Value loss sürekli artıyor | Ödül ölçeği | Ödül bileşenlerinin histogramına bak |

- Her ödül bileşeni ayrı `StatsRecorder` anahtarıyla loglanır.
- **Ödül birim testleri (EditMode, sentetik `StepContext`):**
  - Teğet boyunca v hızıyla gidiş: R = w_v·v/50
  - Geri gidiş: R < 0
  - Kapıda salınım: w_cp yalnızca 1 kez verilir
  - Durma: ödül ≈ 0; 80. kararda −0.5 (Stuck)
  - PurePursuit ile bir tur: toplam ödül > 0

**BenchmarkRunner:**
- Editor'de, ONNX modeli atanmış olarak ve `timeScale = 20` ile C0.10 protokolünü koşturur. `N = 20` ajan, `EvalGrid`, `maxLaps = 3`.
- Sonucu `BenchmarkRecorder` aracılığıyla `evaluator: "unity-inproc"` olarak JSON'a yazar.

**DoD:**
- En az 1 tohum (hedef 3) için `completion_rate ≥ 0.95`.
- `baseline_mlagents.json` yazılmış. İçinde T_ref, `env_config_hash` ve eğitim eğrisi özetleri bulunur.
- ONNX dosyaları `benchmarks/models/` altına kopyalanmış.
- **Env Freeze ilan edilmiş**: RewardConfig ve hash `contracts.md`'ye işlenmiş.
