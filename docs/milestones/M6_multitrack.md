# M6: Çoklu Pist ve Usulü Pist Üretimi

> **Bağımlılıklar:** M1–M5 (commit `fddea58`). Sözleşme: [`../contracts.md`](../contracts.md) C0.20.
> **Amaç:** Ajanın genelleme yeteneğini ölçmek ve artırmak için birden çok pisti (elle tasarlanmış ve tohumlu usulü) aynı ortamda, **Benchmark pistini (Track_A) ve C0.10 protokolünü bozmadan** çalıştırmak.
> **Kural:** Her adımdan sonra durulur, rapor verilir ve komut beklenir. DoD'daki tüm maddeler geçmeden aşama tamamlanmış sayılmaz.

## Değişmezler

- Track_A'nın `env_config_hash` değeri `90240ee2b1a58b5b` olarak kalır (C0.14). `TrackDefinition`'a eklenen alanlar kanonik JSON'a yalnızca varsayılan dışı değer aldıklarında girer.
- Track_A geometrisi ve kurulan collider'lar M5 ile bit düzeyinde aynıdır. Altın parmak izleri (`TrackFingerprint`): geometri `f6930a41d26c9e30`, fiziksel `4ee36c524bd1cf90` (`TrackAGoldenTests`).
- C0.8 adım sırası, `RewardCalculator`, `TerminationPolicy`, `CheckpointTracker` ve `ObservationSpec` (15 ışın, 26 boyut) değişmez.
- ML-Agents sahnesi ve M2 baseline'ına dokunulmaz. Pist seçimi yalnızca köprü ve Watch sahnelerine eklenir.
- Pist süreç başına sabittir (`-trackName` / `-trackIndex`). RESET sırasında pist değişimi yoktur.
- Regresyon: Yeni build ile Track_A değerlendirmesi M5 sonuçlarını tur tur aynen vermelidir: özel PPO 39.18 / 38.68 / 40.50, ML-Agents 41.28 / 41.22 / 41.08.

## Adımlar

1. **Altın testler, `TrackLayout`, `TrackCatalog`, doğrulayıcı profilleri.**
   - `TrackLayout`: Düzlük ve yay zinciri; flex düzlüklerle 2×2 kapanma çözümü; kontrol noktası bake'i.
   - `TrackCatalog.asset`: 0. indeks = Track_A; eklemeler yalnızca sona yapılır.
   - `TrackValidator.CheckProfile`: evrensel kurallar ve profil kuralları.
2. **Track_B (Technical), Track_C (Speedway), `ProceduralTrackGenerator`.** `proc:<seed>` adlandırması; PCG32 ile kabul/ret döngüsü.
3. **Track_D (Hill).** Kot profili, dikeyde uzatılmış duvar collider'ları, `MeshStrip` yol collider'ı.
4. **Köprü.** `-trackName` / `-trackIndex`, HELLO pist alanları, CONFIG `expected_track_id`, `UNKNOWN_TRACK` / `TRACK_MISMATCH` hataları, Watch sahnesi, build ve Track_A regresyonu.
5. **Python.**
   - `track_catalog.json`, `UnityVecEnv(track=)`, `MultiUnityVecEnv(tracks=)`
   - `--track` bayrağı: `train_ppo`, `evaluate`, `watch`
   - mock testleri
   - M5 `best.pt` ile sıfır atış genelleme raporu (`benchmarks/M6_TRACKS.md`)

## Pist profilleri

| Pist | Profil | Hedef |
|---|---|---|
| Track_A | Benchmark | Değişmez (L ≈ 1144 m, W = 12 m) |
| Track_B | Technical | L ≈ 1000–1200 m; en az 8 viraj, 1 saç tokası, 2 şikan; en uzun düzlük ≤ 150 m; W = 11 m |
| Track_C | Speedway | L ≈ 1300 m; en uzun düzlük ≥ 250 m; R_min ≥ 60 m; W = 13 m |
| Track_D | Elevation | Kot 0 → +10 m; eğim ≤ %6; W = 12 m |
| `proc:<seed>` | Procedural | Yalnızca evrensel kurallar |

**Evrensel kurallar** (`TrackValidator.CheckProfile`):
- 600 ≤ L ≤ 1600 m (3 tur 300 s'ye sığmalı, C0.3)
- R_min ≥ 12 m
- Öz-ayrım > W + 4
- |koordinat| < 500
- 10 ≤ W ≤ 14 m
- Toplam dönüş ±360° ± 5°

**Tasarım notu:** Catmull-Rom eğrisi, düzlükten yaya geçişte eğriliği biraz aşar. Ölçülen R_min tasarım yarıçapının yaklaşık 0.93 katıdır (Track_A: R70 → 64.7, R15 → 14.0). Layout yarıçapları bu payla seçilir.

## DoD

- [ ] EditMode:
  - Track_A altın testleri ve `EnvConfigHash_MatchesFrozenValue`
  - Katalogdaki her pist evrensel ve profil kurallarını geçer
  - Pist başına hash dondurulur
  - `proc:<seed>` tekrarlanabilirdir
  - CLI ayrıştırması ve HELLO alanları doğrulanır
- [ ] PlayMode, her pist için:
  - PurePursuit duvara değmeden 3 tur tamamlar
  - Aynı tohumla iki taze ortam bit düzeyinde aynı yörüngeyi verir
  - Track_D'de, 50 m içinde duvar varken hiçbir ışın 1.0 okumaz
- [ ] Python: mock ile pytest (`--track`, çoklu pist, uyuşmazlık hataları) ve gerçek build ile pist başına duman testi.
- [ ] Track_A regresyonu (yukarıdaki değerler, tur tur aynı).
- [ ] M5 `best.pt` ile yeni pistlerde sıfır atış raporu.
- [ ] contracts C0.20 güncel.
