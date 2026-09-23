# M6 Devir Teslim: Çoklu Pist ve Usulü Pist Üretimi

> **Okuma sırası:** Önce bu dosyayı, sonra [`docs/milestones/M6_multitrack.md`](docs/milestones/M6_multitrack.md) planını ve DoD'yi, en son [`docs/contracts.md`](docs/contracts.md) **C0.20** bölümünü oku. Bağlayıcı sayılar, hash'ler ve kurallar contracts'tadır.
> **Tarih:** 2026-09-23. **Dal:** `GS`. **Son commit'ler:** Watch sahnesi `d3f2def`, Adım 1 `09145ed`, Adım 2 `4395397`, Adım 3 `77fe15f`, devir teslim `0f2c6c3`. Adım 4 çalışma ağacında tamamlandı (commit kullanıcı komutuyla atılacak).

## 1. Çalışma kuralları (kullanıcı tercihleri)

- **Dil:** Kullanıcıyla Türkçe konuş. Kod ve kod yorumları İngilizce, dokümanlar Türkçe.
- **Adım disiplini:** Her adımdan sonra dur, raporla ve kullanıcının komutunu bekle. Plan onaylansa bile kullanıcı açıkça "başla" demeden uygulamaya geçme. Commit'i yalnızca kullanıcı isteyince at; başlığı genellikle kullanıcı verir.
- **Commit atfı:** Her commit mesajı `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` satırıyla biter.
- **Değişmezler (C0.14, C0.20):**
  - Track_A `env_config_hash` = `90240ee2b1a58b5b`.
  - Track_A altın parmak izleri: geometri `f6930a41d26c9e30`, fiziksel `4ee36c524bd1cf90`.
  - Gözlem düzeni değişmez (15 ışın, `OBS_DIM` 26, `obs_layout_hash` `b40ca79bdba1c2c2`).
  - C0.8 adım sırası, `RewardCalculator`, `TerminationPolicy` ve `CheckpointTracker` değişmez.
  - ML-Agents sahnesine ve M2 baseline'ına dokunulmaz.
- **Kültür:** Makine tr-TR. C# tarafında `InvariantCulture` ve `Ordinal` zorunlu (C0.13).
- **Yollar ve ortamlar:**
  - Proje yolu ASCII değil. Build ve eğitim `D:\RaceAgent` junction'ı üzerinden yapılır.
  - Python ortamı `D:\RaceAgent\.venv` (3.10.11, torch 2.14.0+cpu, `pip install -e python` kurulu).
  - Donanım yalnızca CPU.

## 2. Mevcut durum: Adım 1–4 tamam

**Testler:** EditMode **121/121** (Adım 4 ile +47: `BridgeTrackTests`), PlayMode **20/20** (Track_C taze ortam determinizmi eklendi) yeşil (`TestResults/*_summary.txt`). Python: `pytest` (mock) yeşil, yeni HELLO/CONFIG testleri `tests/test_mock_env.py` içinde.

**Adım 4 özeti (ayrıntı: contracts C0.20 "Köprü"):**
- `-trackName <kimlik | asset adı | proc:seed>` ve `-trackIndex <i>`; Editor için `BridgeDriver.trackOverride` (string). Çözülemeyen seçim HELLO yerine `UNKNOWN_TRACK` (fatal) döndürür.
- HELLO'ya pist alanları eklendi (`PROTOCOL` 1 kaldı). CONFIG'e `expected_track_id` eklendi; uyuşmazlık `TRACK_MISMATCH` verir.
- `Race_Bridge` ve `Race_Watch` sahnelerine katalog bağlandı (sahne başına 2 satırlık diff). Build `Builds/RaceEnv` yeniden alındı; M5 build'inin yedeği `Builds/RaceEnv_M5` (gitignore).
- Gerçek build doğrulaması `python/scripts/m6_bridge_check.py {smoke,regress,determinism}`. Sonuçlar `benchmarks/eval/m6_*.json` dosyalarında:
  - Track_A regresyonu 13/13 bit düzeyinde aynı.
  - Track_D ve Track_A için süreçler arası ve süreç içi (RESET) STATE akışları aynı.

| İndeks | Kimlik | Profil | W | L (m) | Kapı | `env_config_hash` | PurePursuit turları (s) |
|---|---|---|---|---|---|---|---|
| 0 | Track_A | Benchmark | 12 | 1144.1 | 114 | `90240ee2b1a58b5b` | 54.58 / 51.78 / 51.78 |
| 1 | Track_B | Technical | 11 | 1114.8 | 111 | `038a104393cbfb72` | 67.08 / 65.02 / 65.00 |
| 2 | Track_C | Speedway | 13 | 1262.5 | 126 | `0e099647315ed638` | 44.14 / 40.28 / 40.26 |
| 3 | Track_D | Elevation | 12 | 1094.4 | 109 | `dbce4b7f772fc7b4` | 53.96 / 51.86 / 51.86 |
| −1 | `proc:<seed>` | Procedural | 10.5–13.5 | 800–1400 | — | Tohuma bağlı; referanslar C0.20'de | `proc:0..19` temiz tur tamamlıyor |

Geometri ve fiziksel parmak izleri de C0.20 tablosunda ve `TrackFreezeTests` testlerinde dondurulmuştur.

### Kod haritası (`Assets/Racing/Scripts`)

| Dosya | İçerik |
|---|---|
| `Core/Track/TrackDefinition.cs` | Pist verisi. M6 alanları: `profile`, `notes`, `elevation` (`ElevationKey`), `wallColliderExtraBelow/Above`, `roadCollider` (`GroundBox` / `MeshStrip`), `roadColliderMargin`. `HeightAt(u)`. Yeni alanlar yalnızca varsayılan dışıyken hash'e girer. |
| `Core/Track/TrackGeometry.cs` | Kot taşıyan örnekler. 3B tangent; yatay right, yarıçap ve `Project`. |
| `Core/Track/TrackRuntime.cs` | Fiziksel inşa: MeshStrip yol collider'ı ve uzatılmış duvar collider'ları. |
| `Core/Track/TrackValidator.cs` | `Validate`, `CheckProfile` (evrensel kurallar + profil kuralları), `RequiredWallExtension`. |
| `Core/Track/TrackLayout.cs`, `TrackLayouts.cs` | Düzlük/yay zinciri, flex kapanma, bake. B, C ve D düzenleri; `HillDElevation`. |
| `Core/Track/TrackCatalog.cs` + `Config/TrackCatalog.asset` | Sıralı katalog. `Get`, `IndexOf`, `TryResolve(name, out def, out index)`; `proc:<seed>` → index −1. Yeni pistler yalnızca sona eklenir. |
| `Core/Track/ProceduralTrackGenerator.cs` | v1: `Generate(seed)`, `TryParseName`, `Name`. PCG32 akışı `0x70726f63`. |
| `Core/Track/TrackFingerprint.cs` | Bit düzeyinde `Geometry` ve `Physical` parmak izleri. |
| `Editor/TrackAssets.cs` | Menüler: `Racing/Tracks/Bake Track Assets (M6)`, `Setup Track Catalog (M6)`, `Report Tracks (M6)`. |
| `Tests/EditMode/MultiTrackTests.cs`, `Tests/PlayMode/MultiTrackPlayTests.cs` | M6 testleri. |
| `Bridge/CommandLineArgs.cs`, `Bridge/BridgeDriver.cs`, `Bridge/BridgeProtocol.cs` | Adım 4: `TryParseTrackArgs`, `TryResolveTrack`, `RejectTrack` (UNKNOWN_TRACK), `TrackInfo`/`HelloJson`/`ConfigMsg`/`CheckConfig`. |
| `Editor/BuildScript.cs` | Adım 4: `BindTrackCatalog`, menü `Racing/Tracks/Bind Track Catalog To Bridge Scenes (M6)`. |
| `Tests/EditMode/BridgeTrackTests.cs` | Adım 4: CLI, çözümleme, HELLO, CONFIG testleri; donmuş `track_hash` değerleri. |

## 3. Kritik bulgular ve kısıtlar

1. **PhysX eğim determinizmi (Track_D):**
   - **Belirti:** Aynı Unity sürecinde arka arkaya kurulan iki ortam (pist dahil), eğimli zeminde bit düzeyinde aynı sonuç vermez. Sonuç, süreçte daha önce kaç kez statik pist collider'ı kurulduğuna bağlıdır; 4 koşu periyotlu bir A A B B deseni gözlenir.
   - **Elenen nedenler:**
     - Worker thread sayısı: 0 worker ile de aynı desen çıkıyor.
     - Mesh'e özgülük: BoxCollider şeridiyle de aynı desen çıkıyor.
     - Tekerlek temas verileri: ayrışma adımında iki koşuda da aynı; gövde teması yok.
   - **Kapsam:** Düz pistler (A, B, C, düz MeshStrip) etkilenmez.
   - **Kural:**
     - Pist süreç başına bir kez kurulur (`-trackName`); RESET sırasında pist değişmez.
     - Aynı ortamda `RebuildAgents` ile yapılan RESET bit düzeyinde tekrarlanabilir (`TrackD_RebuildAgents_IsBitwiseReproducible`).
     - Taze süreçler aynı geçmişle başladığı için birbirinin aynısı olmalı. Bu, Adım 4'te gerçek build ile doğrulanmalıdır.
     - Aynı süreçte pisti yeniden kuran bir tasarıma (ör. RESET'te pist değiştirme) geçmeden önce bu bulguyu kullanıcıya hatırlat.
2. **Işın ve duvar:**
   - Işınlar yalnızca yaw eksenine göre yatay atılır. Eğimde M1 yüksekliğindeki duvarlar ışınların %6.1'inde kaçırılır (469/7696).
   - Track_D'de duvar collider'ları ±10 m uzatıldı: 0 kaçırma, en büyük sapma 0.21 m.
   - Kotlu her pist, doğrulayıcının `MeshStrip` ve uzatma kurallarını sağlamak zorundadır.
3. **Spline payı:** Catmull-Rom eğrisi, düzlükten yaya geçişte ölçülen R_min'i tasarım yarıçapının ≈ 0.93 katına düşürür. Layout'larda bu pay hesaba katılmalı (saç tokası R ≥ 14–15).
4. **Usulü pist sınırlaması (v1):** Uzun düzlük yok (en uzun 36–149 m). Yüksek hızda genelleme Track_C ile ölçülür. Üretici değişirse `Version` artırılmalı ve C0.20 güncellenmelidir.
5. **Unity MCP davranışı:**
   - Her C# değişikliğinden sonra domain reload olur ve MCP bir süre `Unity not detected` hatası verir.
   - `%LOCALAPPDATA%\Unity\Editor\Editor.log` boşta kalana kadar bekle, önce `Unity_GetConsoleLogs` çağır, sonra `Unity_RunCommand` ile devam et.
   - Testleri çalıştırmak için: `Racing.Editor.TestRunReporter.Run("EditMode")` veya `Run("PlayMode", "<tam test adı>")`. Sonuçlar `TestResults/<Mode>_summary.txt` dosyasına yazılır; bittiğinde `TestResults/<Mode>_status.txt` içeriği `running` olmaktan çıkar.
   - RunCommand betiklerinde `System.Reflection` ve `System.Diagnostics.Stopwatch` kullanılamaz.
6. **`TeleportTo`** WheelCollider'ın iç durumunu sıfırlamaz. Açık RESET'te köprü `RebuildAgents` kullanır (C0.17).

## 4. Kalan işler

### Adım 4: Köprü (C#), build ve regresyon (TAMAMLANDI; aşağıdaki liste kayıt için duruyor)

1. **`Bridge/CommandLineArgs.cs` ve `Bridge/BridgeDriver.cs`:**
   - `-trackName <id|asset adı|proc:seed>` ve `-trackIndex <i>` argümanları. İkisi çelişirse hata verilir. İkisi de yoksa sahnedeki serileştirilmiş pist kullanılır; bugünkü davranış bayt düzeyinde aynı kalır.
   - BridgeDriver'a serileştirilmiş bir `TrackCatalog` referansı ve Editor modu için bir `trackOverride` alanı eklenir.
   - Çözümleme `TrackCatalog.TryResolve` ile yapılır.
2. **`Core/RaceEnvironment.cs`:** `InitializeFromSerialized(n, seed, mode, TrackDefinition trackOverride = null)`. Mevcut `Initialize(track, …)` çağrısına iletir.
3. **HELLO JSON alanları:** `track_id`, `track_index` (proc = −1), `track_length_m`, `track_checkpoints`, `track_half_width`, `track_hash` (yalnızca `ConfigHash.Compute(trackDef)`). Mevcut `env` alanı korunur.
4. **CONFIG:** İsteğe bağlı `expected_track_id` alanı. `strict` modda uyuşmazlık `TRACK_MISMATCH` hatası verir. Bilinmeyen ad `UNKNOWN_TRACK` hatasıyla (fatal) reddedilir.
   - Alanlar yalnızca JSON'a ekleniyor, bu yüzden **`PROTOCOL`/`Version` 1 kalır**.
   - Hata kodları `BridgeProtocol.cs` ve `python/racing_rl/bridge/protocol.py` dosyalarına eklenir.
5. **Sahneler ve build:**
   - `Editor/BuildScript.cs` içinde `SetupBridgeScene` ve `SetupWatchScene` katalog referansını bağlar.
   - Build `Racing.Editor.BuildScript.BuildBridge` ile alınır (MCP üzerinden veya Editor kapalıyken batchmode'da) ve `Builds/RaceEnv/RaceEnv.exe` dosyasını üretir.
6. **Testler:**
   - EditMode: CLI ayrıştırması ve HELLO/CONFIG alanları (`BridgeProtocolTests` benzeri).
   - Python mock: `racing_rl/bridge/mock_unity.py` yeni HELLO alanlarını üretir.
7. **Track_A regresyonu (zorunlu):**
   - Yeni build ile `python -m racing_rl.train.evaluate --policy torch:../benchmarks/models/custom_ppo_s{1,2,3}.pt --train-seed S ...` ve `onnx:../benchmarks/models/mlagents_baseline_s{1,2,3}.onnx` çalıştırılır.
   - Beklenen değerler, tur tur aynı: özel PPO 39.18 / 38.68 / 40.50 s; ML-Agents 41.28 / 41.22 / 41.08 s.
   - Referans dosyalar: `benchmarks/custom_ppo.json`, `benchmarks/mlagents_bridge.json`; izler `benchmarks/eval/traces/*.npz` (`--trace` ile bitsel kıyaslanabilir).
   - Sonuç aynı değilse dur ve raporla.
8. **Track_D süreçler arası determinizm:** Aynı build, `-trackName Track_D` ve aynı tohumla iki taze süreçte RESET ve N adım sabit aksiyonlarla koşturulur. STATE akışı bitsel olarak aynı olmalıdır. `scripts/m5_repro_check.py` fikri yeniden kullanılabilir.

### Adım 5: Python

1. **`python/racing_rl/bridge/track_catalog.json` ve `tracks.py`:**
   - İçerik: kimlik, indeks, dondurulmuş `env_config_hash`, L, kapı sayısı.
   - JSON Unity'deki bir `Racing/Tracks/Export Track Catalog` menüsüyle üretilir. Bir EditMode testi dosyanın güncelliğini denetler.
2. **`UnityVecEnv(track=...)`:**
   - `-trackName` argümanını ekler.
   - `expected_env_hash` açıkça verilmediyse beklenen hash katalogdan seçilir; `proc:*` için None olur ve hash HELLO'dan kaydedilir.
   - HELLO'daki `track_id` doğrulanır ve `env.track_info` olarak sunulur.
3. **`MultiUnityVecEnv(tracks=[...])`:**
   - Süreç k, `tracks[k % len]` pistini koşturur; her süreç kendi hash'iyle doğrulanır.
   - `infos["track_index"]` eklenir.
4. **CLI:**
   - `train_ppo` için `--track` ve YAML'da `run.tracks`.
   - `train.evaluate` ve `train.watch` için `--track`.
   - Checkpoint'e `extra.tracks = {id: hash}` yazılır; karışık eğitimde `env_config_hash` alanına bileşik `multi:<sha>` yazılır. M4 formatı değişmez.
5. **Testler:** mock ile pytest (`--track`, çoklu pist, hash/pist uyuşmazlığı hataları) ve gerçek build ile pist başına kısa duman testi.
6. **Sıfır atış (zero-shot) raporu, `benchmarks/M6_TRACKS.md`:**
   - M5 `custom_ppo_s{1,2,3}.pt` ve ML-Agents ONNX, C0.10 protokolüyle (20 episode, tohumlar 1000..1019, 3 tur, deterministik) Track_B, C, D ve birkaç `proc:` tohumunda değerlendirilir.
   - Tablo: completion ve flying lap. PurePursuit referans turlarıyla da kıyaslanır.
   - 10M karma eğitim **kapsam dışı**; karar sonraya bırakıldı.

Adım 4 ve 5 bittikten sonra M6 DoD kutucukları işaretlenir ve C0.20 "devam ediyor" başlığından çıkarılır.

## 5. Diğer

- **`python/scripts/watch_ediyor.py`:** Kullanıcının canlı izleme betiği; `best.pt`'yi Editor'deki Race_Bridge sahnesinde gerçek zamanlı sürdürür. Çalıştırma: `cd python` ardından `python scripts/watch_ediyor.py [ckpt]`. Genel sürüm: `python -m racing_rl.train.watch`. Editor'de başka pist izlemek için sahnedeki `BridgeDriver.trackOverride` alanına kimlik veya `proc:<seed>` yazılır (`train.watch` Adım 5'teki `--track` bayrağına kadar beklenen hash olarak Track_A'yı kullanır ve başka pisti `ProtocolMismatchError` ile reddeder).
- **Konsol kodlaması:** Proje yolu ASCII değil ve Python konsolu cp1252. Betik çıktılarında tam yolu yazdırmak `UnicodeEncodeError` verir; yalnız dosya adı yazdırılır.
- **Çalışma planı hafızası:** Kullanıcı hafıza dosyaları `C:\Users\27ome\.claude\projects\D--Unity-Projects-Araba-Yar----Ajan-\memory\` altında.

## 6. Yeni ajanın ilk görevi

İlk mesajda aşağıdaki görevi başlat:

> "`M6_HANDOVER.md`, `docs/milestones/M6_multitrack.md` ve `docs/contracts.md` C0.20'yi (özellikle 'Köprü') oku. `git log --oneline -5` ile Adım 4 commit'ini doğrula. Unity MCP ile EditMode (121) ve PlayMode (19) testlerini, `python -m pytest` ile Python testlerini çalıştırıp yeşil olduklarını teyit et. Ardından **M6 Adım 5** için dosya bazında kısa bir uygulama planı (`track_catalog.json` + export menüsü, `UnityVecEnv(track=)`, `MultiUnityVecEnv(tracks=)`, `--track` bayrakları, checkpoint `extra.tracks`, mock testleri, sıfır atış raporu) sun ve kullanıcının 'başla' komutunu bekle."
