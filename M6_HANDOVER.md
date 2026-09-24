# M6 Devir Teslim: Çoklu Pist ve Usulü Pist Üretimi

> **Güncel iş (2026-09-23):** M6'dan sonra UI1 (Unity içi pist seçici ve izleme arayüzü) başladı. Devam noktası [`UI1_HANDOVER.md`](UI1_HANDOVER.md); önce onu oku. Bu dosyanın §1 kuralları ve §3 bulguları geçerliliğini koruyor.

> **Okuma sırası:** Önce bu dosyayı, sonra [`docs/milestones/M6_multitrack.md`](docs/milestones/M6_multitrack.md) planını ve DoD'yi, en son [`docs/contracts.md`](docs/contracts.md) **C0.20** bölümünü oku. Bağlayıcı sayılar, hash'ler ve kurallar contracts'tadır.
> **Tarih:** 2026-09-23. **Dal:** `GS`. **Son commit'ler:** Watch sahnesi `d3f2def`, Adım 1 `09145ed`, Adım 2 `4395397`, Adım 3 `77fe15f`, devir teslim `0f2c6c3`, **Adım 4 `c452638`** (`feat(m6): CLI track flags, bridge hello/config protocol and build verification`). Adım 4 sonrası devir teslim `ec5e4ce`. **Adım 5 çalışma ağacında tamamlandı**; commit kullanıcı komutuyla atılacak.
> **Durum:** **M6 tamamlandı** (DoD'nin tamamı işaretli). Sıradaki iş kullanıcının kararına bağlı, bkz. §4.

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

## 2. Mevcut durum: M6 tamam (Adım 1–5)

**Testler (hepsi yeşil):**
- EditMode **123/123**. Adım 4 ile +47 test (`BridgeTrackTests`), Adım 5 ile +2 test (`TrackCatalogExportTests`) geldi.
- PlayMode **22/22**. Adım 5'te PurePursuit 3 tur testine `proc:1` ve `proc:1000` eklendi.
- Python `pytest` **116/116**: 67 eski test, `tests/test_m6_tracks.py` (mock, 41) ve `tests/test_unity_tracks.py` (gerçek build, `-m unity`, 8). Yalnız slow testi hariç.
- Unity sonuçları `TestResults/*_summary.txt` dosyalarında.

**Adım 5 özeti (ayrıntı: contracts C0.20 "Python (Adım 5)" ve "Sıfır atış (Adım 5)"):**
- **Katalog:** `python/racing_rl/bridge/track_catalog.json`, Unity menüsü `Racing/Tracks/Export Track Catalog (M6)` ile üretilir (yalnız Editor kodu, build değişmedi). EditMode testi JSON'un güncel olduğunu denetler; katalog veya pist değişirse menü yeniden çalıştırılmalıdır.
- **`racing_rl.bridge.tracks`:** `resolve_track`, `load_catalog`, `composite_hash`, `training_tracks`, `UnknownTrackError`.
- **Ortamlar:** `UnityVecEnv(track=)` (`-trackName`, CONFIG `expected_track_id`, HELLO denetimi, `track_info`, `infos["track_index"]`) ve `MultiUnityVecEnv(tracks=)` (süreç k: `tracks[k % len]`). `expected_env_hash` varsayılanı `AUTO`'dur.
- **CLI:** `train_ppo --track` (tekrarlanabilir) / YAML `run.tracks` / `--eval-track`, `train.evaluate --track`, `train.watch --track`.
- **Checkpoint:** `env_config_hash` tek pistte o pistin hash'i, karışık eğitimde `multi:<sha16>`; `extra.tracks = {id: hash}`. Biçim `racing_rl.ppo/v1` değişmedi. `load_policy` artık checkpoint'in eğitim hash'ini katalogla denetliyor (değerlendirilen pistten bağımsız).
- **Regresyon:** Track_A, `m6_bridge_check.py regress` ile yeniden koşuldu (13/13 bit düzeyinde aynı). Sıfır atış betiği de Track_A'yı yeni API'yle koşup M5 ile birebir aynı sonucu aldı (6/6).
- **Sıfır atış (`benchmarks/M6_TRACKS.md`):** 42 değerlendirmenin 20'si bitti. Track_C ve `proc:1000` 6/6, Track_B ve Track_D 0/6. Çarpışmalar dar virajlarda ve Track_D'de yokuşun sonunda kümeleniyor. C0.10 deterministik protokolü pist başına ikili (0 veya 1) sonuç veriyor.

**Build durumu:**
- `Builds/RaceEnv/RaceEnv.exe` Adım 4 koduyla alındı (`build_id` `ee81723cde19439da2497c9fb946fc4c`, gitignore). `c452638`'deki player koduyla aynıdır; sonradan değişen tek dosya bir PlayMode testidir ve build'e girmez.
- M5 build'inin yedeği `Builds/RaceEnv_M5` klasöründe (`build_id` `1c6af42e1ed2481fbbb15a49da198c82`).
- Adım 5'te C# player kodu değişmezse yeniden build gerekmez. Yalnız Editor menüsü eklemek (ör. katalog export) build'i etkilemez.
- Build alınırsa: `Racing.Editor.BuildScript.BuildBridge` (MCP, yaklaşık 40 s). Ardından `python scripts/m6_bridge_check.py regress` ile Track_A regresyonu tekrarlanmalıdır.

**Adım 4 özeti (ayrıntı: contracts C0.20 "Köprü"):**
- `-trackName <kimlik | asset adı | proc:seed>` ve `-trackIndex <i>`. Editor için `BridgeDriver.trackOverride` (string). Çözülemeyen seçim HELLO yerine `UNKNOWN_TRACK` (fatal) döndürür ve süreç 2 koduyla çıkar.
- HELLO'ya pist alanları eklendi, `PROTOCOL` 1 kaldı. CONFIG'e `expected_track_id` eklendi; uyuşmazlık `TRACK_MISMATCH` verir.
- `Race_Bridge` ve `Race_Watch` sahnelerine katalog bağlandı.
- Gerçek build doğrulaması `python/scripts/m6_bridge_check.py {smoke,regress,determinism}` ile yapıldı; sonuçlar `benchmarks/eval/m6_*.json` dosyalarında:
  - smoke: 16/16.
  - Track_A regresyonu: 13/13 koşu, metrikler ve izler M5 ile bit düzeyinde aynı.
  - Track_D ve Track_A: süreçler arası ve süreç içi (RESET) STATE akışları bit düzeyinde aynı, sabit aksiyonlarla da M5 politikasıyla da.
- M6 DoD'de işaretlenenler: EditMode, PlayMode, Track_D, Track_A regresyonu. Açık kalanlar: Python, sıfır atış raporu, C0.20'nin kapatılması (hepsi Adım 5).

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
     - Taze süreçler aynı geçmişle başladığı için birbirinin aynısıdır. Adım 4'te gerçek build ile doğrulandı (`benchmarks/eval/m6_track_determinism.json`).
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

### Adım 4: Köprü (C#), build ve regresyon: TAMAMLANDI (`c452638`)

Özet §2'de, bağlayıcı kurallar contracts C0.20 "Köprü" bölümünde.

### Adım 5: Python entegrasyonu ve sıfır atış raporu: TAMAMLANDI

Özet §2'de. Aşağıdaki liste kayıt için duruyor.

**Başlangıç noktası (Adım 4 sonrası Python durumu):**
- `UnityVecEnv(..., extra_args=[...])` ile `-trackName` bugün de elle geçirilebiliyor. HELLO pist alanları `env.hello` içinde geliyor.
- CONFIG henüz `expected_track_id` göndermiyor. Beklenen hash varsayılanı Track_A'dır (`FROZEN_ENV_CONFIG_HASH`); başka pist için `expected_env_hash` açıkça verilmezse `ProtocolMismatchError` alınır.
- `racing_rl.bridge.multi_env.MultiUnityVecEnv` K süreç × N ajan olarak zaten var (M5); yalnız pist desteği eklenecek.
- Protokol sabitleri `protocol.py` dosyasında: `ERR_UNKNOWN_TRACK`, `ERR_TRACK_MISMATCH`, `HELLO_TRACK_FIELDS`.
- Mock (`mock_unity.py`) `track_id`/`track_index` parametreleri alıyor ve `expected_track_id` denetliyor.
- `python/scripts/m6_bridge_check.py` gerçek build kontrollerinde örnek olarak kullanılabilir (`raw_config`, `run_process`).

**Yapılacaklar:**
1. **Pist kataloğu (Python):** `python/racing_rl/bridge/track_catalog.json` ve `tracks.py`.
   - İçerik: kimlik, indeks, dondurulmuş `env_config_hash`, `track_hash`, L, kapı sayısı, W.
   - JSON Unity'deki `Racing/Tracks/Export Track Catalog (M6)` menüsüyle üretilir (yalnız Editor kodu, build gerekmez). Bir EditMode testi JSON'un katalogla güncel olduğunu denetler.
   - `proc:<seed>` katalogda yoktur. Beklenen hash None'dır ve HELLO'dan kaydedilir.
2. **`UnityVecEnv(track=...)`:**
   - `-trackName` argümanını kendisi ekler ve CONFIG'e `expected_track_id` koyar.
   - `expected_env_hash` açıkça verilmediyse katalogdan seçilir.
   - HELLO'daki `track_id` doğrulanır ve `env.track_info` olarak sunulur.
   - `track=None` bugünkü davranıştır (Track_A); M5 kodu değişmeden çalışmalıdır.
3. **`MultiUnityVecEnv(tracks=[...])`:**
   - Süreç k, `tracks[k % len]` pistini koşturur ve kendi hash'iyle doğrulanır.
   - `infos["track_index"]` (ajan başına) eklenir.
   - Pist süreç başına sabittir; RESET'te değişmez (C0.20 eğim bulgusu).
4. **`--track` bayrağı (CLI):**
   - `train_ppo`: `--track` (tekrarlanabilir) ve YAML'da `run.tracks`. Tek pistte checkpoint'in `env_config_hash` alanı o pistin hash'idir. Karışık eğitimde bileşik `multi:<sha>` yazılır ve `extra.tracks = {id: hash}` eklenir. M4 checkpoint formatı değişmez.
   - `train.evaluate --track` ve `train.watch --track` (Editor'de `trackOverride` ile birlikte kullanılır). Pist Track_A değilse checkpoint yüklemede hash denetimi, politikanın eğitildiği pistin hash'iyle yapılmalıdır (`load_policy` bugün `FROZEN_ENV_CONFIG_HASH` kullanıyor).
5. **Testler:**
   - Mock ile pytest: `--track`, çoklu pist, hash ve pist uyuşmazlığı hataları, `track=None` geriye uyumluluğu.
   - Gerçek build ile pist başına kısa duman testi: A–D ve bir `proc:` tohumu.
6. **Sıfır atış (zero-shot) genelleme benchmark'ı, `benchmarks/M6_TRACKS.md`:**
   - Modeller: M5 `custom_ppo_s{1,2,3}.pt` ve ML-Agents `mlagents_baseline_s{1,2,3}.onnx`.
   - Protokol: C0.10 (20 episode, tohumlar 1000..1019, EvalGrid, 3 tur, deterministik μ). Pistler: Track_B, C, D ve birkaç `proc:` tohumu (ör. 0, 1, 7, 1000).
   - Tablo: completion rate, flying lap medyanı, bitiş nedenleri; PurePursuit referans turlarıyla kıyas. Track_A satırı referans olarak M5 değerleridir.
   - Ön gözlem (Adım 4 determinizm koşusu): `custom_ppo_s1` Track_D'de tur tamamlayamadan çarpıyor. Track_A'da aynı koşuda 41 tur attı.
   - 10M karma eğitim **kapsam dışıdır**; karar sonraya bırakıldı.
7. **Kapanış:** M6 DoD'deki kalan üç kutu (Python, sıfır atış raporu, contracts C0.20 güncel) işaretlenir. C0.20 "devam ediyor" başlığından çıkarılır.

### Sonraki olası işler (kullanıcı kararı; hiçbiri başlatılmadı)

1. **10M karma eğitim** (`train_ppo --track ...`): altyapı hazır ve mock ile test edildi. Rapor, eğitim dağılımına dar ve ardışık virajlı pistlerin (Track_B, `proc:` tohumları) ve kotlu pistlerin eklenmesini öneriyor. Doğrulama şu an yalnız ilk pistte yapılıyor; çok pistli model seçimi ayrı bir karardır.
2. **Genelleme protokolü:** C0.10 (deterministik μ, ±0.5 m / ±2°) pist başına ikili sonuç veriyor. Daha çeşitli başlangıçlar veya daha çok pist ile ek bir protokol düşünülebilir. C0.10'un kendisi değişmemeli (M2–M5 kıyasları ona bağlı).
3. **Eğim algısı:** Gözlemde kot veya eğim bilgisi yok (C0.6). Track_D bulgusu bunu bir hipotez olarak gösteriyor; gözlem düzeni değişirse `obs_layout_hash` ve tüm dondurulmuş hash'ler değişir (Env Freeze).

## 5. Diğer

- **`python/scripts/watch_ediyor.py`:** Kullanıcının canlı izleme betiği; `best.pt`'yi Editor'deki Race_Bridge sahnesinde gerçek zamanlı sürdürür. Çalıştırma: `cd python` ardından `python scripts/watch_ediyor.py [ckpt]`. Genel sürüm: `python -m racing_rl.train.watch`. Editor'de başka pist izlemek için sahnedeki `BridgeDriver.trackOverride` alanına kimlik veya `proc:<seed>` yazılır ve `python -m racing_rl.train.watch --track <aynı kimlik>` çalıştırılır. `--track` verilmezse beklenen pist Track_A'dır ve başka pist `ProtocolMismatchError` ile reddedilir. `watch_ediyor.py`'ye dokunulmadı (kullanıcı kuralı); yalnız Track_A ile çalışır.
- **Konsol kodlaması:** Proje yolu ASCII değil ve Python konsolu cp1252. Betik çıktılarında tam yolu yazdırmak `UnicodeEncodeError` verir; yalnız dosya adı yazdırılır.
- **Çalışma planı hafızası:** Kullanıcı hafıza dosyaları `C:\Users\27ome\.claude\projects\D--Unity-Projects-Araba-Yar----Ajan-\memory\` altında.

## 6. Yeni ajanın ilk görevi

İlk mesajda aşağıdaki görevi başlat:

> "`M6_HANDOVER.md`, `benchmarks/M6_TRACKS.md` ve `docs/contracts.md` C0.20'yi oku. `git log --oneline -5` ile M6 Adım 5 commit'ini doğrula. Unity MCP ile EditMode (123) ve PlayMode (22) testlerini, `python -m pytest` ile Python testlerini (116) çalıştırıp yeşil olduklarını teyit et. M6 tamamlandı; §4'teki 'Sonraki olası işler' listesini kullanıcıya özetle ve hangisiyle devam edileceğine dair kararını bekle."
