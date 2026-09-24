# RaceAgent: Otonom Yarış Ajanı

Unity'de fizik tabanlı bir yarış ortamı ve bu ortamda sıfırdan yazılmış PyTorch PPO ile eğitilen sürücü ajanı. Önce ortam ve ödül hazır ML-Agents PPO ile doğrulandı (baseline), sonra aynı ortam özel TCP köprüsüyle kendi PPO'muza bağlandı ve kıyaslandı.

## Durum (2026-09-24)

Planlanan aşamaların hepsi tamamlandı:

| Aşama | İçerik |
|---|---|
| M1 | Pist, WheelCollider aracı, geometrik checkpoint, raycast sensörleri |
| M2 | Ödül tasarımı ve ML-Agents PPO baseline, **Env Freeze** |
| M3 | İkili TCP lockstep köprüsü + Gymnasium / vektör ortam |
| M4 | Sıfırdan PyTorch continuous PPO |
| M5 | Eğitim döngüsü, SPS optimizasyonu, kıyas |
| M6 | Çoklu pist (Track_A–D) ve usulü pist üretimi (`proc:<seed>`) |
| UI1 | Unity içi pist seçici ve izleme arayüzü (`Race_Viewer.unity`) |

**Sonuçlar (Track_A, 20 episode, 3 tur, deterministik):** Özel PPO flying lap tohum medyanı **39.18 s**, ML-Agents **41.22 s** (%4.95 daha hızlı), completion her tohumda 1.00. Eğitim hızı ~14k karar/s (ML-Agents ~3k); 1e7 karar ≈ 14 dk. Yalnız Track_A'da eğitilen modeller yeni pistlerde sıfır atışta 42 değerlendirmeden 20'sini bitirdi; dar virajlı Track_B ve Track_D'yi hiçbiri bitiremedi. Sıradaki iş kullanıcı kararına bağlı.

## Teknolojiler

- **Unity 6000.4.6f1** (C#, PhysX, eski Input Manager, UI Toolkit). `com.unity.ml-agents` 4.1.0; `com.unity.ai.inference` 2.6.1 `unity/Packages/` altında gömülü ve yamalı (bkz. `EMBEDDED_PATCH.md`).
- **Python 3.10.11**, iki ortam:
  - `python/.venv`: özel motor. torch 2.14.0+cpu, gymnasium, numpy 2, onnxruntime, tensorboard. `pip install -e python`.
  - `python/.venv-mla`: baseline. `mlagents==1.1.0` (`python/baselines/mlagents/requirements.lock.txt`).
- Donanım yalnız CPU (AMD GPU, CUDA yok). Paralellik: süreç başına N ajan × K Unity süreci.

## Yapı

Üç domain; ayrıntı `ARCHITECTURE.md`. Diller arası yolların tek kaynağı kökteki `paths.json` (C# `Racing.Editor.RepoPaths`, Python `racing_rl.paths`); yeni kodda yol sabit yazılmaz.

```
paths.json                 build, çıktı ve katalog yolları
unity/                     Unity projesi (projectPath = D:\RaceAgent\unity)
  Assets/Racing/Scripts/
    Core/       Ortam çekirdeği: araç, pist, sensör, ödül, sonlanma, RaceEnvironment. ML-Agents referansı yasak.
    MLAgents/   M2 baseline ajanı (Race_MLAgents sahnesi)
    Bridge/     M3 TCP köprüsünün Unity tarafı (Race_Bridge sahnesi, outputs/builds/RaceEnv)
    Viewer/     UI1 izleme arayüzü
    Editor/     Sahne/pist kurulum, build ve test menüleri (Racing/...), RepoPaths
  Assets/Racing/Tests/{EditMode,PlayMode}
python/
  racing_rl/
    bridge/     protokol, UnityVecEnv / MultiUnityVecEnv, pist kataloğu (track_catalog.json)
    rl/         PPO: ağlar, GAE, buffer, normalizasyon, checkpoint
    train/      train_ppo, evaluate, watch, compare CLI'ları
    paths.py    paths.json okuyucu
  configs/            PPO ve köprü ayarları
  scripts/            m3/m5/m6 doğrulama ve benchmark betikleri
  tests/              pytest (tests/rl: PPO birim testleri)
  baselines/mlagents/ ML-Agents baseline config, bağımlılıklar, rapor betiği
outputs/
  benchmarks/   versiyonlanan sonuçlar: JSON'lar, eval/, eval/traces/, models/ (.pt, .onnx), plots/
  builds/       player build'leri (RaceEnv, RaceEnv_MLA)          [gitignore]
  runs/         eğitim koşuları, Unity logları, mlagents/ sonuçları [gitignore]
  test-results/ Unity test özetleri                               [gitignore]
```

## Ortam sözleşmesi (özet)

- **Zaman:** fizik 50 Hz (`simulationMode = Script`, Enhanced Determinism açık), karar 10 Hz (K = 5 action repeat). En çok 3000 karar/episode.
- **Aksiyon:** `[steer, throttle_brake] ∈ [-1,1]²`. AWD, hız sınırı 45 m/s, geri vites yok.
- **Gözlem:** 26 float: 15 raycast (180°, 50 m), ileri/yanal hız, yaw hızı, iki bakış açısının sin/cos'u (0 ve 30 m), yanal sapma, önceki aksiyon, yere değen teker oranı.
- **Adım sırası:** `ApplyAction` → `Physics.Simulate(0.02)` → ajanlar sırayla `AfterPhysicsStep` (ödül ve sonlanma).
- **Sonlanma:** Wall, WrongWay, Stuck (8 s), Flip, OutOfBounds, PhysicsError; TimeLimit ve Finished truncated sayılır.
- **Benchmark protokolü:** EvalGrid başlangıcı, tohumlar 1000..1019, 3 tur, ortalama aksiyon (μ), metrik 2.–3. turların flying lap medyanı. Sayılar her zaman taze süreçten alınır (`-trackName`), Viewer ölçüm aracı değildir.

## Değişmezler (Env Freeze)

Bunlar EditMode testleriyle kilitlidir. Değişirlerse baseline ve regresyonlar geçersiz olur.

- Track_A `env_config_hash` **`90240ee2b1a58b5b`**, `obs_layout_hash` **`b40ca79bdba1c2c2`**.
- Track_A parmak izleri: geometri `f6930a41d26c9e30`, fiziksel `4ee36c524bd1cf90`.
- Gözlem düzeni, adım sırası, `RewardCalculator`, `TerminationPolicy`, `CheckpointTracker` ve `RewardConfig.asset` değişmez. ML-Agents sahnesine dokunulmaz.
- Pist ekleme/değiştirme sonrası `Racing/Tracks/Export Track Catalog (M6)` menüsüyle katalog yeniden üretilir.

Koddaki `C0.x` etiketleri silinmiş `docs/contracts.md` bölümlerine atıftır; gerekirse git geçmişinden okunabilir (`git show b56a362:docs/contracts.md`).

## Çalışma kuralları

- Kullanıcıyla Türkçe konuş. Kod ve kod yorumları İngilizce, dokümanlar Türkçe.
- Plan onaylansa bile kullanıcı "başla" demeden uygulamaya geçme. Commit'i yalnız kullanıcı isteyince at.
- Makine tr-TR: C#'ta `InvariantCulture` ve `Ordinal` zorunlu. Core'da `Time.*` ve `UnityEngine.Random` yasak (`DeterministicRng` kullan), adım başına GC ayırması yok.
- Proje yolu ASCII değil. Build, eğitim ve Python işleri `D:\RaceAgent` junction'ı (repo kökü) üzerinden yapılır; Unity projesi `D:\RaceAgent\unity`.
- Batchmode test için Editor kapalı olmalı; Editor açıkken Unity MCP araçları kullanılır. Test sonuçları `outputs/test-results/`.

## Sık komutlar (`D:\RaceAgent\python` içinden)

```
python -m racing_rl.train.train_ppo --config configs/ppo_parity.yaml --track Track_A --run-dir ../outputs/runs/demo
python -m racing_rl.train.evaluate --policy torch:../outputs/benchmarks/models/custom_ppo_s2.pt --train-seed 2 --track Track_A --out ../outputs/runs/eval.json
python -m racing_rl.train.watch --track proc:7
pytest                 # varsayılan "not slow"; -m unity gerçek build (outputs/builds/RaceEnv) ister
```

`outputs/` altında yalnız `benchmarks/` versiyonlanır. Build'ler `Racing/...` Editor menülerinden üretilir.
