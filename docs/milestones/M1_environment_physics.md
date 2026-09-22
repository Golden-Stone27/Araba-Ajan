# M1: Ortam ve Fizik Mimarı (Unity/C#)

> **Alt ajan talimatı:** Bu aşamayı uygulamak için yalnızca bu dosya ve [`../contracts.md`](../contracts.md) yeterlidir.
> Sözleşmelerdeki (C0.x) bir değeri değiştirmen gerekirse, değiştirmeden önce gerekçesiyle raporla.
> **Bağımlılıklar:** Yok (ilk aşama).
> **Teslim edilen (sonraki aşamaya):** M2: `Racing.Core` derlemesi, `Sandbox.unity`, pist ve araç konfigürasyonları, `EpisodeSignals`, geçen EditMode/PlayMode testleri, hesaplanmış `obs_layout_hash`.
> **Kural:** DoD'daki tüm maddeler geçmeden aşama tamamlanmış sayılmaz.

**Kapsam:**
- Pist üretimi, araç dinamiği, checkpoint takibi, sensörler, gözlem oluşturucu, klavye ve otopilot girdisi.
- M2 için `EpisodeSignals` üretimi.
- `SimulationDriver` "Manual" modu: FixedUpdate içinde `ApplyAction(input)` ve ardından `env.PhysicsStep()`.

**Dosyalar:**
- `Core/Track/{TrackDefinition.cs (ScriptableObject), TrackRuntime.cs : ITrack, CatmullRom.cs}`
- `Editor/TrackBuilder.cs`
- `Core/Vehicle/{VehicleConfig.cs, VehicleController.cs, AntiRollBar.cs}`
- `Core/Sensors/{RaySensor.cs, ObservationSpec.cs, ObservationBuilder.cs}`
- `Core/Episode/{CheckpointTracker.cs, LapTimer.cs, EpisodeMonitor.cs, DeterministicRng.cs}`
- `Core/Input/{KeyboardInputSource.cs, PurePursuitInputSource.cs}`
- `Core/{RaceAgentCore.cs, RaceEnvironment.cs, SimulationDriver.cs, SimConfig.cs, ConfigHash.cs}`
- `Core/Debug/{RaceHud.cs (OnGUI), RaceGizmos.cs}`
- `Scenes/Sandbox.unity`

**Pist (TrackBuilder):**
- Orta çizgi, kapalı merkezcil Catmull-Rom eğrisiyle (α = 0.5) kurulur. 1 m aralıklarla örneklenir. Her örnek için konum P[k], teğet T[k], sağ vektör N[k] = norm(cross(up, T)) ve kümülatif s[k] saklanır.
- Genişlik W = 12 m. Pist düz bir zemin üzerindedir (y = 0).
- Zemin, üst yüzü y = 0 olan tek büyük bir BoxCollider'dır (`Road` katmanı). Yol mesh'i yalnızca görseldir ve collider'ı yoktur.
- **Duvarlar:** 2 m'lik BoxCollider zinciri. Boyut (1.0 kalınlık, 1.5 yükseklik, 2.2 uzunluk; %10 bindirme). Konum `P ± N·(W/2 + 0.5)`. Katman `Wall`.
- **Checkpoint'ler:** Her 10 m'de bir `Checkpoint{Index, P(s), T, N, W/2, s}`. Start/finish çizgisi s = 0'dadır.
- Pist merkezi orijine yakın olmalıdır (|p| < 500 m, float hassasiyeti için).
- **Validator (EditMode testi):**
  - Pist uzunluğu L ∈ [900, 1400] m
  - Minimum eğrilik yarıçapı R_min ≥ 12 m
  - Kendini kesme yok: komşu olmayan örnekler arasındaki mesafe > W + 2
  - En az 1 firkete (R 12–20), 1 şikan, 1 düzlük (≥ 200 m), 1 hızlı viraj (R 60–80). Hem sol hem sağ viraj olmalı.
- **Başlangıç kontrol noktaları** (x, z; öneridir, validator geçmezse düzeltilir):
  `(0,0) (150,0) (250,0) (300,30) (310,90) (280,140) (220,150) (190,120) (160,100) (120,120) (90,100) (60,130) (30,180) (-30,190) (-80,160) (-90,100) (-60,60) (-70,20) (-40,0)`. Yaklaşık 1.1 km eder.

**Araç (VehicleConfig başlangıç değerleri, M1 ayarlar):**
- Rigidbody:
  - Kütle 1200 kg, COM (0, −0.4, 0.1)
  - `linearDamping` 0.05, `angularDamping` 0.05
  - Gövde BoxCollider (1.8, 0.6, 4.2), y = 0.55, katman `Car`
- WheelCollider:
  - Yarıçap 0.34, kütle 20, `suspensionDistance` 0.2, `forceAppPointDistance` 0.1
  - Yay 35000, sönüm 4500, `targetPosition` 0.5. Köşe kütlesi 300 kg için ζ = c / (2√(k·m)) ≈ 0.69.
  - İleri sürtünme: (extSlip 0.4, extVal 1, asySlip 0.8, asyVal 0.5, stiffness 1.5)
  - Yanal sürtünme: (0.2, 1, 0.5, 0.75, 1.5)
- Viraj denge çubuğu: 5000 N/birim yay farkı, ön ve arka akslar.
- Aksiyon eşlemesi C0.5'teki gibidir.
- **TeleportTo:** `linearVelocity` ve `angularVelocity` sıfırlanır. Poz `rb.position/rotation` ve transform ile ayarlanır. Tekerlerde motor, fren ve steer değerleri 0 yapılır. `rotationSpeed = 0` yazılır; bu özellik yazılamazsa ilk alt adımda `brakeTorque = float.MaxValue` uygulanır. Ardından `Physics.SyncTransforms()` çağrılır ve iç direksiyon durumu ile son aksiyon sıfırlanır.
- **Spawn yüksekliği:** Aracın 25 adım boyunca 0 aksiyonla bekledikten sonra dikey hızı < 0.05 m/s olacak şekilde ayarlanır ve sabitlenir.

**Girdi:**
- `KeyboardInputSource`: `Input.GetAxis("Horizontal")` direksiyon, `"Vertical"` gaz ve fren.
- `PurePursuitInputSource` (otomatik testler için referans sürücü):
  - İleri bakış mesafesi L_d = clamp(0.8·v, 8, 25) m
  - Direksiyon = atan2(2·L_wb·sin α, L_d) / δ_max
  - Hedef hız v* = min(45, √(8·R(s + L_d)))
  - Gaz = clamp(0.5·(v* − v), −1, 1)

**EpisodeMonitor:**
- Her fizik adımında şu sinyalleri üretir: `WallContact` (collision tamponundan), `Flipped` (zamanlayıcılı), `OutOfBounds`, `NoProgressTimeout`, `NonFinite`, `Cp`.
- `WallClearance = W/2 − |lateral| − 0.9` (analitik, raycast gerektirmez).
- **Karar vermez.** Ödül ve sonlanma kararı M2'nin sorumluluğundadır. M1'de `RewardCalculator` yoktur; `AfterPhysicsStep` Reward = 0 döndürür ve sinyalleri loglar.

**Hata yakalama stratejileri:**

| Risk | Önlem |
|---|---|
| Yüksek hızda teker titremesi veya araç batması | Substeps ayarı, kütle oranı ve ζ ≈ 0.7 |
| Takla | Düşük COM, viraj denge çubuğu, downforce |
| Duvarı delip geçme | 1 m kalın duvar kutuları ve `ContinuousDynamic` |
| Raycast'in araca veya trigger'a çarpması | Yalnızca `Wall` maskesi ve `QueryTriggerInteraction.Ignore` |
| Checkpoint kaçırma veya çift sayma | Trigger yerine segment-düzlem kesişim testi |
| Script modunda yanlış poz | `Physics.SyncTransforms()` çağrısı, interpolasyon kapalı |
| NaN | `VehicleState.IsFinite` kontrolü ve PhysicsError sinyali |
| Kültür hatası | InvariantCulture (C0.13) |
| GC kaynaklı takılma | Profiler'da `PhysicsStep` için GC Alloc = 0 B |

**Testler ve DoD:**

1. **EditMode, pist validator'ı:** Tüm kriterler geçer. Checkpoint sayısı `floor(L/10)` olur ve s değerleri artan sıradadır.
2. **EditMode, `CheckpointTracker` (sentetik konumlarla):**
   - İleri geçiş sayacı 1 artırır.
   - Ters geçiş WrongWay döndürür.
   - Kapı atlama (teleport) sayılmaz.
   - Tüm kapılar sırayla geçilince LapCompleted döner.
   - Aynı kapıda ileri-geri salınım yalnızca 1 kez sayılır.
3. **PlayMode, RaySensor:**
   - 10 m önde duvar varken merkez ışın (i = 7) `0.2 ± 0.002` ölçer.
   - Duvar yoksa değer `1.0` olur.
   - Işın yolundaki trigger collider yok sayılır.
4. **PlayMode, düz hat testi:**
   - `a = (0, 1)` ile 5 s sürüş: hız monoton artar, yanal kayma < 0.5 m, araç devrilmez.
   - 0→100 km/h süresi 4–7 s arasındadır.
   - 100 km/h'den tam frenle duruş mesafesi ≤ 45 m.
5. **PlayMode, otopilot:**
   - PurePursuit 3 turu duvara değmeden tamamlar.
   - Tüm kapılar sırayla algılanır.
   - Tur süreleri kaydedilir (alt referans). Turlar arası fark < %0.5.
6. **PlayMode, fuzz testi:** 16 ajan, 20k adım, rastgele aksiyon.
   - Tüm gözlemler sonlu ve C0.6 aralıklarında.
   - PhysicsError = 0.
   - `WallContact` sinyali > 0 kez tetiklenir (tespitin çalıştığını gösterir).
7. **PlayMode, determinizm ön testi:** Aynı tohum ve betikli aksiyonlarla 1000 adım iki kez koşulur. Tüm konumlar bit düzeyinde eşit olmalıdır.
8. **Manuel:**
   - Sandbox'ta klavyeyle stabil tur atılır.
   - HUD şunları gösterir: km/h, 26 gözlem değeri, sıradaki checkpoint, tur saati, son/en iyi tur ve sinyaller.
   - Gizmos ışınları, orta çizgiyi ve kapıları çizer.

---

## Uygulama durumu (2026-09-23)

M1 tamamlandı; DoD 1–7 otomatik testlerle geçti (EditMode 14/14, PlayMode 6/6). Plandan sapmalar ve sonraki aşamaları bağlayan notlar `contracts.md` → **C0.15** bölümündedir (prefab yerine çalışma zamanı fabrikaları, genişletilmiş `ITrack`/`EpisodeSignals`, eval spawn s = 1 m). DoD 8'in klavye kısmı insan tarafından doğrulanmalıdır.
