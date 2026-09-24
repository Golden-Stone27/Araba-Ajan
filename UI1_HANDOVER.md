# UI1 Devir Teslim: Unity İçi Pist Seçici ve İzleme Arayüzü (Faz 1)

> **Okuma sırası:** Önce bu dosya, sonra [`M6_HANDOVER.md`](M6_HANDOVER.md) §1 (çalışma kuralları) ve §3 (kritik bulgular), en son [`docs/contracts.md`](docs/contracts.md) C0.14 ve C0.20.
> **Tarih:** 2026-09-23. **Dal:** `GS`. **Commit'ler:** Adım 1 `a8932c7` ("Adım 1: Viewer altyapısı"), Adım 2 `03b540c` ("Adım 2 Bitti"). Çalışma ağacı bu dosya dışında temiz.
> **Durum:** UI1 Faz 1 tamamlandı (Adım 1–3). Adım 3 kapanışı: [`docs/milestones/UI1_viewer.md`](docs/milestones/UI1_viewer.md) ve contracts C0.21. Bekleyen görev yok; sonraki faz kullanıcı kararıdır.

## 1. Çalışma kuralları (özet; tamamı M6_HANDOVER §1)

- Kullanıcıyla Türkçe konuş. Kod ve kod yorumları İngilizce, dokümanlar Türkçe.
- Plan onaylansa bile kullanıcı açıkça "başla" demeden uygulamaya geçme. Her adımdan sonra dur, raporla, komut bekle.
- Commit'i yalnız kullanıcı isteyince at (kullanıcı genelde kendisi atıyor). Commit mesajı `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` ile biter.
- **Değişmezler:** Track_A `env_config_hash` `90240ee2b1a58b5b`, geometri parmak izi `f6930a41d26c9e30`, fiziksel parmak izi `4ee36c524bd1cf90`. Gözlem düzeni, C0.8 adım sırası, ödül ve bitiş mantığı değişmez. ML-Agents sahnesine dokunulmaz.
- tr-TR makine: C# tarafında `InvariantCulture` ve `Ordinal` zorunlu.
- Yollar: çalışma `D:\RaceAgent` junction'ı üzerinden. Unity Editor gerçek yoldan (`D:\Unity Projects\Araba Yarışı Ajanı`) açık.

## 2. UI1 Faz 1 hedefi ve kararlar

Terminalden betik çalıştırmak yerine Unity içinde, fareyle kullanılan hafif bir arayüz. ONNX model seçimi bu fazın kapsamı dışında.

- **Kullanıcı kararları (onaylı):** Arayüz UI Toolkit ile yapılır; uGUI paketi eklenmez, `Packages/manifest.json` temiz kalır. Otopilot (PurePursuit) varsayılan olarak açık.
- **Mimari:** Pist değişince sahne yeniden yüklenmez. `ViewerController` eski `RaceEnvironment`'ı siler (`SetActive(false)` + `Destroy`). Sahnedeki pasif şablonu (`RaceEnvironment (Template)`, `initializeOnAwake = false`) klonlar, klonu viewer'ın sahnesine taşır ve `InitializeFromSerialized(1, 1000, EvalGrid, def)` çağırır. Pist `TrackCatalog.TryResolve` ile çözülür, köprüyle aynı yol.
- **Core, Bridge ve MLAgents koduna dokunulmadı.** Yeniden build gerekmedi.
- **Ölçüm aracı değil:** Aynı süreçte pist yeniden kurulduğu için Track_D (eğimli) bit düzeyinde tekrarlanabilir değildir (C0.20 bulgusu). Benchmark sayıları her zaman taze süreçten (`-trackName`) alınır (C0.21).

## 3. Mevcut durum (Adım 1 + 2)

### Dosya haritası

| Dosya | İçerik |
|---|---|
| `Assets/Racing/Scripts/Viewer/Racing.Viewer.asmdef` | Yalnız `Racing.Core`'a referans. autoReferenced: player build'lerine derlenir ama köprü sahnesi kullanmaz. |
| `Viewer/ViewerController.cs` | `LoadTrack(name, out error)`, `ResetCar()`, `Autopilot`, `ShortcutsEnabled`, `ElapsedSeconds`, olay `CarSpawned`. Sürüş döngüsü `FixedUpdate`'te: `ApplyAction` → `PhysicsStep` (C0.8). P ve R tuşları. |
| `Viewer/ViewerChaseCamera.cs` | Core `ChaseCamera` ile aynı çerçeve; `CarSpawned` olayında `Snap()`. |
| `Viewer/ViewerUI.cs` | Sol üst panel: `DropdownField` (A–D ve `Usulü (proc)`), tohum `TextField` (yalnız rakam), `Yükle`, `Rastgele`, `Otopilot (P)` toggle, `Sıfırla (R)`, bilgi ve hata satırları. Testler için public `SelectTrack`, `LoadSeed`, `RandomSeed`. |
| `Viewer/ViewerHud.cs` | Sağ alt: mod etiketi (`OTOPİLOT`/`KLAVYE`), hız (km/s), `Süre mm:ss.ff`. `FormatCentis` public. |
| `Viewer/Editor/ViewerSetup.cs` (+ `Racing.Viewer.Editor.asmdef`, namespace `Racing.Editor`) | Menü `Racing/Viewer/Create Viewer Scene (UI1)`: `Race_Viewer.unity` ve `Assets/Racing/UI/ViewerPanelSettings.asset` üretir. |
| `Assets/Racing/UI/` | `Viewer.uss`, `ViewerTheme.tss` (`unity-theme://default`), `ViewerPanelSettings.asset` (1600×900, ScaleWithScreenSize). |
| `Assets/Racing/Scenes/Race_Viewer.unity` | Build Settings'e eklenmedi; yalnız Editor Play modu için. |
| `Assets/Racing/Tests/PlayMode/ViewerPlayTests.cs` | 10 test (5 altyapı, 5 arayüz). |

**Mevcut dosyalardaki tek değişiklik:** `Racing.Tests.PlayMode.asmdef`'e eklenen `Racing.Viewer` referansı. Nedeni: `TestRunReporter` yalnız `Racing.Tests.PlayMode` derlemesini koşturuyor.

### Davranış kararları

- **Süre sınırı:** 300 s'lik TimeLimit viewer'da yok sayılır, otopilot tur atmaya devam eder.
- **Otomatik reset:** Kazada otopilotlu araç `ResetAgent(0)` ile yeniden başlar. Klavyedeki araç yalnız takla, pist dışı veya sayısal hatada sıfırlanır (Sandbox ile aynı).
- **R ve Sıfırla:** `RebuildAgents()` kullanır, araç yeniden kurulur (C0.17: ışınlama WheelCollider durumunu korur).
- **Bellek:** Proc `TrackDefinition`'lar `HideFlags.DontSave` taşır ve `UnloadUnusedAssets` tarafından boşaltılmaz; pist değişince viewer onları açıkça siler. Eski pistin mesh ve materyalleri bir kare sonra `Resources.UnloadUnusedAssets()` ile temizlenir.
- **Odak:** Kontroller kullanıldıktan sonra odağı bırakır (`Blur`), böylece ok tuşları ve WASD arayüzü gezmez. Tohum alanı odaktayken P ve R kapalıdır.

### Testler (Adım 2 sonrası, hepsi yeşil)

- EditMode **123/123**.
- PlayMode **32/32**: 22 eski ve 10 viewer testi.
- pytest bu fazda koşulmadı; Python tarafında değişiklik yok. Son değer M6'dan **116/116**.
- Play modunda `ScreenCapture.CaptureScreenshot` ile gözle doğrulandı: Track_A, Track_D, proc:42 ve hata satırı.

### Bu fazda öğrenilenler

- `Object.Instantiate`, klonu **aktif sahneye** koyar. Viewer, klonu `SceneManager.MoveGameObjectToScene` ile kendi sahnesine taşır. Aksi halde sahne kapanınca ortam geride kalır; testler bunu yakaladı.
- `EditorSceneManager.NewScene(..., Additive)`, kaydedilmemiş adsız bir sahne açıkken hata verir. `ViewerSetup` bu durumu ve açık ve temiz viewer sahnesini yerinde yeniden kurarak çözer. Dosyanın üzerine yazar (silmez), böylece GUID korunur. Kaydedilmemiş değişiklik varsa reddeder.
- Unity 6.4'te eskimiş API'ler: `FindObjectsByType(..., FindObjectsSortMode)` ve `SceneHandle`'dan `int`'e örtük dönüşüm. Yerine `FindObjectsByType<T>(FindObjectsInactive)` ve `var` kullanılır.
- UI Toolkit testlerinde `resolvedStyle` yerine satır içi `style.display.value` denetlenir; panel güncellemesine bağlı kalınmaz.
- Editörde önceki oturumlardan kalan `DontSave` proc tanımları olabilir (ör. `proc:7`). Testler yalnız kendi ürettikleri nesneleri denetler.
- Unity MCP: her C# değişikliğinden ve Play modu girişinden sonra `Unity not detected` hatası gelir. `%LOCALAPPDATA%\Unity\Editor\Editor.log` boşta kalana kadar bekle, `Unity_GetConsoleLogs` çağır, sonra devam et. Bash'te `find`/`Glob` ile `Assets` taraması zaman aşımına uğrayabilir; doğrudan yol kullan.
