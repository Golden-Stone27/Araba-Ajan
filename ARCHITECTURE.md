# RaceAgent Mimarisi

Repo üç domaine ayrılır: **`unity/`** simülasyonu çalıştırır, **`python/`** öğrenir ve ölçer, **`outputs/`** ikisinin ürettiği her şeyi tutar. Diller arası tek bağ kökteki `paths.json`'dur.

## Sistem mimarisi

```
                           paths.json (repo kökü)
                  ┌────────────────┴────────────────┐
       Racing.Editor.RepoPaths (C#)         racing_rl.paths (Python)

unity/ ──Racing/… build──► outputs/builds/RaceEnv/RaceEnv.exe ◄──başlatır── racing_rl.bridge.UnityVecEnv
                                   │   TCP lockstep: HELLO → CONFIG → READY,          │
                                   └── sonra RESET / STEP ⇄ STATE, CLOSE ─────────────┘

racing_rl.train (PPO) ──► outputs/runs/<koşu>/ (checkpoint, metrik, Unity logları)
python/scripts/        ──► outputs/benchmarks/ (resmi JSON, trace, model, grafik)

python/.venv-mla mlagents-learn ⇄ outputs/builds/RaceEnv_MLA ──► outputs/runs/mlagents/
                                     └─ make_baseline_report.py ──► outputs/benchmarks/baseline_mlagents.json

unity/ Export Track Catalog ──► python/racing_rl/bridge/track_catalog.json
unity/ TestRunReporter / -runTests ──► outputs/test-results/
```

- **Ortam çekirdeği** (`Racing.Core`): `RaceEnvironment` deterministik adımı yürütür (50 Hz fizik, 10 Hz karar). Bunu üç sürücü kullanır: `BridgeDriver` kendi PPO'muz için, `MlaSimulationDriver` ML-Agents baseline'ı için, `ViewerController` de izleme arayüzü için.
- **Köprü:** TCP sunucusu Python tarafındadır. `UnityVecEnv` bir port açar, `RaceEnv.exe`'yi `-trackName` gibi argümanlarla başlatır ve `HELLO` mesajındaki `env_config_hash` / `obs_layout_hash` değerlerini dondurulmuş hash'lerle karşılaştırır.
- **Env Freeze:** Hash'ler dosya yolundan değil içerikten hesaplanır. Klasör taşımak onları değiştirmez, `Core/` içindeki davranışı değiştirmek değiştirir.

## Dizin düzeni

| Dizin | Sorumluluk |
|---|---|
| `paths.json` | Build, çıktı ve katalog yollarının tek kaynağı. İki dil de buradan okur. |
| `unity/` | Kendi başına açılabilen Unity projesi (`Assets/`, `Packages/`, `ProjectSettings/`). `Library/`, `Logs/`, `UserSettings/` gitignore'dadır. |
| `unity/Assets/Racing/Scripts/Core` | Ortam çekirdeği. ML-Agents referansı yasaktır. |
| `unity/Assets/Racing/Scripts/{Bridge,MLAgents,Viewer}` | Çekirdeği sürüten üç adaptör. |
| `unity/Assets/Racing/Scripts/Editor` | Sahne/pist kurulumu, build, test raporlayıcı ve `RepoPaths`. |
| `python/racing_rl/` | Kurulabilir paket: `bridge` (protokol, vektör ortam), `rl` (PPO), `train` (CLI'lar), `paths`. |
| `python/configs/` | PPO ve köprü ayarları. Yol içermez. |
| `python/scripts/` | Milestone doğrulama ve benchmark betikleri. |
| `python/tests/` | pytest testleri. `-m unity` gerçek build ister, `-m slow` uzun eğitimleri koşar. |
| `python/baselines/mlagents/` | ML-Agents baseline config'i, kilitli bağımlılıklar ve rapor betiği. |
| `outputs/benchmarks/` | Versiyonlanan sonuçlar. `outputs/` altında git'e giren tek klasör budur. |
| `outputs/{builds,runs,test-results}/` | Player build'leri, eğitim koşuları ve test özetleri. Hepsi silinip yeniden üretilebilir. |

## Giriş noktaları

1. `unity/Assets/Racing/Scripts/Core/RaceEnvironment.cs`: adım sırası, ajan yaşam döngüsü, `env_config_hash`.
2. `python/racing_rl/bridge/vec_env.py`: Unity sürecini başlatır ve protokolü konuşur.
3. `python/racing_rl/train/train_ppo.py`: eğitim döngüsü, checkpoint ve doğrulama.

## Runbook

Tüm komutlar ASCII junction üzerinden çalışır (proje yolu ASCII değil):

```
mklink /J D:\RaceAgent "<repo yolu>"
```

**Python ortamları** (`D:\RaceAgent` içinden):

```
py -3.10 -m venv python\.venv
python\.venv\Scripts\python -m pip install -e "python[test,train]" --extra-index-url https://download.pytorch.org/whl/cpu
py -3.10 -m venv python\.venv-mla
python\.venv-mla\Scripts\python -m pip install -r python\baselines\mlagents\requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**Unity:** Hub'da `Add` ile `D:\RaceAgent\unity` eklenir. Build'ler `Racing/...` menülerinden alınır ya da Editor kapalıyken batch ile:

```
Unity.exe -batchmode -quit -projectPath D:\RaceAgent\unity -executeMethod Racing.Editor.BuildScript.BuildBridge
Unity.exe -batchmode -projectPath D:\RaceAgent\unity -runTests -testPlatform EditMode -testResults D:\RaceAgent\outputs\test-results\EditMode.xml
```

**Eğitim ve ölçüm** (`D:\RaceAgent\python` içinden, `.venv` etkin):

```
python -m racing_rl.train.train_ppo --config configs/ppo_parity.yaml --track Track_A --run-dir ../outputs/runs/demo
python -m racing_rl.train.evaluate --policy torch:../outputs/benchmarks/models/custom_ppo_s2.pt --train-seed 2 --track Track_A --out ../outputs/runs/eval.json
python -m racing_rl.train.watch --track proc:7
pytest
python scripts/m6_bridge_check.py regress
```

**ML-Agents baseline** (`D:\RaceAgent` içinden, `.venv-mla` etkin):

```
mlagents-learn python/baselines/mlagents/config/race_ppo.yaml --env outputs/builds/RaceEnv_MLA/RaceEnv.exe --results-dir outputs/runs/mlagents --run-id baseline_s1
```
