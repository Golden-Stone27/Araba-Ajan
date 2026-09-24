# UI1: Unity İçi Pist Seçici ve İzleme Arayüzü (Faz 1)

**Durum:** Adım 1 (`a8932c7`, altyapı) ve Adım 2 (`03b540c`, UI Toolkit paneli ve HUD) tamamlandı. Adım 3 (kapanış) bu dokümanla tamamlandı. **UI1 Faz 1 tamamlandı.**

> **Bağımlılıklar:** M1–M6. Sözleşme: [`../contracts.md`](../contracts.md) C0.21.
> **Amaç:** Terminalden betik çalıştırmadan, Unity içinde fareyle pist seçip aracı izlemek.
> **Kapsam dışı:** ONNX veya checkpoint seçimi (Faz 2), standalone viewer build.

## Kararlar (kullanıcı onaylı)

- **Arayüz teknolojisi:** UI Toolkit (Unity 6'da yerleşik). uGUI paketi eklenmedi, `Packages/manifest.json` değişmedi.
- **Sürüş:** Otopilot (PurePursuit) varsayılan olarak açık. P ile klavyeye geçilir.
- **Kod sınırı:** Core, Bridge ve MLAgents koduna dokunulmaz. Yeni kod ayrı bir `Racing.Viewer` derlemesinde durur.

## Mimari

- **Pist değişimi:** Pist değişince sahne yeniden yüklenmez. `ViewerController` eski `RaceEnvironment`'ı siler ve sahnedeki pasif şablonu (`initializeOnAwake = false`) klonlar. Klon viewer sahnesine taşınır ve `InitializeFromSerialized(1, 1000, EvalGrid, def)` ile kurulur; pist mesh'leri, collider'lar ve araç yeniden oluşur.
- **Pist çözümleme:** Köprüyle aynı yol, `TrackCatalog.TryResolve`. Track_A–D ve `proc:<seed>` desteklenir.
- **Başlangıç:** Araç EvalGrid ile çizginin 1 m ilerisine konur, sabit tohum 1000.
- **Adım sırası:** C0.8 (`ApplyAction` → `PhysicsStep`).
- **Kamera:** `ViewerChaseCamera` her doğuşta aracın arkasına hemen geçer.
- **Arayüz:** `ViewerUI` sol üstte pist seçici, `ViewerHud` sağ altta mod, km/s ve süre gösterir.

Dosya haritası ve davranış kararlarının ayrıntısı: [`../../UI1_HANDOVER.md`](../../UI1_HANDOVER.md) §3.

## Kullanım

1. Gerekirse sahneyi üret: menü `Racing/Viewer/Create Viewer Scene (UI1)`. Açık ve kaydedilmemiş değişikliği olmayan viewer sahnesi yerinde yeniden kurulur.
2. `Assets/Racing/Scenes/Race_Viewer.unity` sahnesini aç ve Play'e bas.
3. Açılır listeden pist seç. `Usulü (proc)` seçilince tohum alanı açılır: `Yükle`, `Rastgele` ya da Enter.
4. Kısayollar: **P** otopilot/klavye, **R** aracı çizgiye yeniden kur. Klavye sürüşü ok tuşları veya WASD ile yapılır.

## DoD

- [x] Pist seçici: A–D ve `proc:<seed>`; sahne yeniden yüklenmeden pist mesh'i ve collider'lar güncellenir.
- [x] Araç seçilen pistin başlangıç çizgisinde doğar; takip kamerası aracı izler.
- [x] HUD: anlık hız (km/s) ve geçen süre.
- [x] Pist değişiminde bellek temizlenir: proc tanımları ve eski mesh'ler.
- [x] Core, Bridge ve Env Freeze değişmedi. Track_A `env_config_hash` `90240ee2b1a58b5b` viewer testinde de doğrulandı.
- [x] Testler: EditMode 123/123, PlayMode 32/32 (10'u viewer testi).
- [x] Play modunda gözle doğrulama: ekran görüntüleri ve kullanıcının elle yaptığı testler, görünür hata yok.
- [x] Sözleşme notu C0.21.

## Doğrulama notu

Adım 3'te testler yeniden koşulmadı. Adım 2'de yeşil geçen kod ile şu anki kod aynıdır (arada yalnız doküman değişti). UI1 boyunca Python ve köprü build'i değişmedi; pytest'in son değeri M6'dan 116/116'dır.

## Bilinen durum

- Duvarlar Play modunda koyu veya siyah görünüyor (görsel mesh normalleri, `TrackRuntime.BuildWallSide`). Kullanıcı kararıyla ele alınmadı; fiziği etkilemez.

## Sonraki fazlar (kullanıcı kararı)

- Faz 2: Arayüzden ONNX veya checkpoint seçip politikayı izlemek.
- İsteğe bağlı: standalone viewer build.
