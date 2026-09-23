# M6 Sıfır Atış Raporu: M5 Modelleri Yeni Pistlerde

> **Soru:** Yalnız Track_A'da eğitilmiş M5 modelleri, hiç görmedikleri pistlerde ne kadar sürebiliyor?
> **Protokol:** C0.10 (EvalGrid, tohumlar 1000..1019, 3 tur, deterministik μ, ajan başına ilk episode), köprü değerlendiricisi (`racing_rl.train.evaluate`, `UnityVecEnv(track=...)`).
> **Modeller:** özel PPO `custom_ppo_s{1,2,3}.pt` (M5 `best.pt`) ve ML-Agents `mlagents_baseline_s{1,2,3}.onnx` (M2). Hepsi yalnız Track_A'da eğitildi. **Build:** `Builds/RaceEnv` (`build_id` `ee81723cde19439da2497c9fb946fc4c`, Unity 6000.4.6f1). Üretim: `python scripts/m6_zero_shot.py run` ve `report`; ham sonuçlar `benchmarks/eval/m6_zero_shot.json`.

## Özet

- **Kapsam:** 7 yeni pist × 6 model = 42 değerlendirme (her biri 20 episode). **20 değerlendirme 3 turu bitirdi.** Özel PPO 10/21, ML-Agents 10/21.
- **Pist bazında:** Track_C (Speedway) ve `proc:1000` 6/6 model ile bitirildi. `proc:7` 5/6, `proc:1` 2/6, `proc:0` 1/6 modelle bitirildi. **Track_B (Technical) ve Track_D (Elevation) 0/6:** hiçbir model tek bir tur bile tamamlayamadı; iki pistte de 120 episode'un hepsi `wall` ile bitti.
- **Ya hep ya hiç:** 48 değerlendirmenin hepsinde completion ya 1.00 ya 0.00. C0.10 deterministik μ kullanır ve pertürbasyon küçüktür (±0.5 m, ±2°). Bu yüzden bir modelin 20 ajanı aynı yolu izler ve çoğunlukla aynı metrede (±1 m) çarpar. Pist başına fiilî örneklem 6 modeldir; completion oranları bu gözle okunmalıdır.
- **Bitirilen pistlerde hız:**
  - Özel PPO her ortak pistte ML-Agents'tan hızlıdır: Track_C 39.60 s'ye karşı 44.12 s (−10.2 %), `proc:7` 36.95 s'ye karşı 38.58 s, `proc:1000` 33.60 s'ye karşı 36.00 s. Track_A'da fark −4.95 % idi.
  - `proc:1000`'de T/PP oranı (özel 0.756, ML 0.810) Track_A ile neredeyse aynıdır (0.757 / 0.796). Model bir pisti bitirebildiğinde hız kaybı yoktur.
  - Track_C'de T/PP oranı yüksektir (0.983 / 1.096). Bunun nedeni modelin yavaşlığı değildir (ortalama hız 30.5 m/s, Track_A'da 28.1 m/s). PurePursuit bu pistte kendisi hızlıdır: viraj yarıçapı en az 84 m, PP en yüksek hızı 157 km/h. Oran pist karakterine bağlıdır ve pistler arasında dikkatle karşılaştırılmalıdır.
- **Başarısızlık nedeni dar virajlar:** Çarpışmalar pistin en dar virajlarında kümelenir. Bir pistte R < 30 m olan toplam uzunluk arttıkça onu bitiren model sayısı düşer (tablo aşağıda):

| Pist | R_min (m) | R < 30 m uzunluğu (m) | Dar viraj (R < 30 m) | 60 m içinde ters yönlü dar viraj çifti | Bitiren model (özel · ML) |
|---|---|---|---|---|---|
| Track_C | 83.9 | 0 | 0 | 0 | 3 · 3 |
| `proc:1000` | 18.4 | 40 | 4 | 1 | 3 · 3 |
| `proc:7` | 16.7 | 45 | 3 | 0 | 2 · 3 |
| `proc:1` | 16.6 | 47 | 3 | 1 | 1 · 1 |
| Track_A (eğitim pisti) | 14.0 | 67 | 3 | 1 | 3 · 3 |
| `proc:0` | 12.2 | 74 | 8 | 2 | 1 · 0 |
| Track_D | 21.8 | 102 | 8 | 1 | 0 · 0 |
| Track_B | 13.9 | 218 | 14 | 4 | 0 · 0 |

  Ölçüm: Unity `TrackGeometry`, 1 m örnekler. Dar viraj, R < 30 m olan kesintisiz örnek dizisidir. Çift, 60 m içinde art arda gelen ve yönü ters iki dar virajdır.

- **Track_B:**
  - 4/6 model s ≈ 657–662 m'deki saç tokasında çarpıyor: pistin en dar yeri, R ≈ 14–15 m. Saç tokasına R 25 m'lik bir viraj ve yaklaşık 40 m'lik bir düzlükten sonra giriliyor.
  - 2/6 model s ≈ 182–213 m'de, yarıçapı giderek daralan virajda (R 45 → 29 → 21 m) çarpıyor.
  - Track_A'da da R ≈ 14 m'lik bir saç tokası var. Fark, dar bölümlerin miktarı ve dizilişidir: R < 30 m uzunluğu Track_A'nın 3.3 katı, 14 dar viraj ve 4 ters yönlü çift var.
- **Track_D:**
  - 120 episode'un 75'i (%63) kotlu bölümde bitiyor. 71'i yokuşun son 60 m'sinde (s ≈ 425–480 m, kot 9.3–10 m): burada eğim %2.4'ten sıfıra iniyor, önce R 40 m'lik bir viraj, tepede de yön değiştiren bir S geliyor. 4'ü platoda (s ≈ 482–594 m).
  - 45 episode (%37) inişten sonraki düz bölümde, R 35 m → R 24–28 m S-virajında (s ≈ 1004–1019 m) bitiyor.
  - Model bazında: özel s1, ML s2 ve s3 tepede; özel s3 düz S'de; özel s2 ve ML s1 ikisinde de.
  - Track_D'nin R_min'i (21.8 m) Track_A'dakinden büyük. R 40 m'lik bir virajda çarpmak, düz pistlerdeki dağılıma uymuyor. Bu yüzden yokuş bölgesinin payı var. Olası açıklama (doğrulanmadı): gözlemde eğim veya kot bilgisi yok (C0.6), düz pistte öğrenilen fren noktaları yokuşta geçerli değil.
- **Tohumlar:**
  - Özel s3 en sağlam modeldir (7 yeni pistin 5'i).
  - Track_A'nın en hızlısı olan özel s2 (38.68 s) en zayıf genelleyendir (2/7): Track_A başarımı genellemeyi öngörmüyor.
  - ML-Agents: s1 4/7, s2 ve s3 3/7.
- **Kapsam dışı ve öneriler:** 10M karma eğitim kapsam dışıdır; karar kullanıcıya bırakıldı. Yapılırsa bu rapor şunları önerir:
  1. Eğitim dağılımına dar ve ardışık virajlı bölümler (Track_B benzeri ve `proc:` tohumları) ile kotlu pistler eklenmeli.
  2. Genelleme ölçümü için C0.10'a ek olarak daha çeşitli başlangıçlar (farklı kapılar, daha büyük pertürbasyon) veya daha çok pist kullanılmalı. Mevcut protokol pist başına ikili bir sonuç veriyor.
  3. Model seçimi yalnız Track_A doğrulamasıyla yapılmamalı.

## Pistler

| Pist | Profil | L (m) | W (m) | Kapı | Kot | PurePursuit turları (s) | PP flying (s) | `env_config_hash` |
|---|---|---|---|---|---|---|---|---|
| Track_A | Benchmark | 1144.1 | 12 | 114 | — | 54.58 / 51.78 / 51.78 | 51.78 | `90240ee2b1a58b5b` |
| Track_B | Technical | 1114.8 | 11 | 111 | — | 67.08 / 65.02 / 65.00 | 65.01 | `038a104393cbfb72` |
| Track_C | Speedway | 1262.5 | 13 | 126 | — | 44.14 / 40.28 / 40.26 | 40.27 | `0e099647315ed638` |
| Track_D | Elevation | 1094.4 | 12 | 109 | +10 m | 53.96 / 51.86 / 51.86 | 51.86 | `dbce4b7f772fc7b4` |
| proc:0 | Procedural | 1169.2 | 11.5 | 116 | — | 57.72 / 55.88 / 55.90 | 55.89 | `00724614a3c71752` |
| proc:1 | Procedural | 1267.5 | 12 | 126 | — | 59.04 / 56.22 / 56.24 | 56.23 | `860ce2361684922d` |
| proc:7 | Procedural | 1063.5 | 12.5 | 106 | — | 47.62 / 45.16 / 45.16 | 45.16 | `59966c9f08fc1027` |
| proc:1000 | Procedural | 971.7 | 12.5 | 97 | — | 47.00 / 44.44 / 44.44 | 44.44 | `7b4353dd8647e3d3` |

## Kafa kafaya (3 eğitim tohumu)

Completion: 60 episode üzerinden (3 tohum × 20); parantezde en az bir episode'u bitiren tohum sayısı. T: bu tohumların flying lap medyanlarının medyanı (C0.10 T_ref tanımı). T/PP: aynı pistte PurePursuit flying lap'ine oran (Track_A satırı referanstır). Mesafe: bitiremeyen episode'ların sonlanmadan önce sürdüğü tur (medyan).

| Pist | Completion özel | Completion ML | T özel (s) | T ML (s) | T/PP özel | T/PP ML | 1. tur özel / ML (s) | Ort. hız özel / ML (m/s) | Mesafe özel / ML (tur) |
|---|---|---|---|---|---|---|---|---|---|
| Track_A | 1.00 (3/3) | 1.00 (3/3) | 39.18 | 41.22 | 0.757 | 0.796 | 42.68 / 44.32 | 28.13 / 26.73 | — / — |
| Track_B | 0.00 (0/3) | 0.00 (0/3) | — | — | — | — | — / — | 22.71 / 23.48 | 0.59 / 0.59 |
| Track_C | 1.00 (3/3) | 1.00 (3/3) | 39.60 | 44.12 | 0.983 | 1.096 | 43.06 / 47.46 | 30.53 / 27.92 | — / — |
| Track_D | 0.00 (0/3) | 0.00 (0/3) | — | — | — | — | — / — | 26.45 / 23.80 | 0.43 / 0.40 |
| proc:0 | 0.33 (1/3) | 0.00 (0/3) | 41.60 | — | 0.744 | — | 44.42 / — | 26.56 / 22.48 | 0.56 / 0.20 |
| proc:1 | 0.33 (1/3) | 0.33 (1/3) | 44.00 | 44.90 | 0.783 | 0.799 | 47.38 / 48.07 | 26.51 / 25.04 | 0.66 / 0.69 |
| proc:7 | 0.67 (2/3) | 1.00 (3/3) | 36.95 | 38.58 | 0.818 | 0.854 | 40.28 / 42.44 | 27.57 / 26.42 | 0.93 / — |
| proc:1000 | 1.00 (3/3) | 1.00 (3/3) | 33.60 | 36.00 | 0.756 | 0.810 | 36.86 / 39.27 | 27.44 / 25.88 | — / — |

## Bitiş nedenleri (60 episode = 3 tohum × 20)

| Pist | Özel PPO | ML-Agents |
|---|---|---|
| Track_A | finished 60 | finished 60 |
| Track_B | wall 60 | wall 60 |
| Track_C | finished 60 | finished 60 |
| Track_D | wall 60 | wall 60 |
| proc:0 | wall 40, finished 20 | wall 60 |
| proc:1 | wall 40, finished 20 | wall 40, finished 20 |
| proc:7 | finished 40, wall 20 | finished 60 |
| proc:1000 | finished 60 | finished 60 |

## Çarpışma konumu

Bitiremeyen ilk episode'ların sonlandığı konum (s, start çizgisinden metre). Konumlar 15 m içinde art arda geliyorsa aynı kümeye girer; küme başına medyan s ve episode sayısı.

| Pist | Bitiremeyen özel / ML | Kümeler özel | Kümeler ML |
|---|---|---|---|
| Track_A | 0 / 0 | — | — |
| Track_B | 60 / 60 | ≈660 m ×40, ≈182 m ×20 | ≈658 m ×40, ≈213 m ×20 |
| Track_C | 0 / 0 | — | — |
| Track_D | 60 / 60 | ≈425 m ×29, ≈1004 m ×26, ≈594 m ×3, ≈456 m ×1, ≈482 m ×1 | ≈430 m ×41, ≈1010 m ×19 |
| proc:0 | 40 / 60 | ≈164 m ×20, ≈1150 m ×20 | ≈230 m ×37, ≈196 m ×20, ≈45 m ×3 |
| proc:1 | 40 / 40 | ≈799 m ×20, ≈866 m ×20 | ≈875 m ×40 |
| proc:7 | 20 / 0 | ≈988 m ×20 | — |
| proc:1000 | 0 / 0 | — | — |

**Track_D kot bölgeleri** (C0.20 profili; bölge sınırları yaklaşıktır):

| Bölge | s (m) | Özel PPO | ML-Agents |
|---|---|---|---|
| düz (başlangıç) | 0–150 | 0 | 0 |
| yokuş çıkışı | 150–480 | 30 | 41 |
| plato | 480–600 | 4 | 0 |
| iniş | 600–960 | 0 | 0 |
| düz (bitiş) | 960–L | 26 | 19 |

## Tohum bazında matris

Hücre: completion · flying lap medyanı (s).

| Pist | Özel s1 | Özel s2 | Özel s3 | ML s1 | ML s2 | ML s3 |
|---|---|---|---|---|---|---|
| Track_A | 1.00 · 39.18 | 1.00 · 38.68 | 1.00 · 40.50 | 1.00 · 41.28 | 1.00 · 41.22 | 1.00 · 41.08 |
| Track_B | 0.00 · — | 0.00 · — | 0.00 · — | 0.00 · — | 0.00 · — | 0.00 · — |
| Track_C | 1.00 · 41.68 | 1.00 · 39.46 | 1.00 · 39.60 | 1.00 · 44.37 | 1.00 · 44.12 | 1.00 · 43.56 |
| Track_D | 0.00 · — | 0.00 · — | 0.00 · — | 0.00 · — | 0.00 · — | 0.00 · — |
| proc:0 | 0.00 · — | 0.00 · — | 1.00 · 41.60 | 0.00 · — | 0.00 · — | 0.00 · — |
| proc:1 | 0.00 · — | 0.00 · — | 1.00 · 44.00 | 1.00 · 44.90 | 0.00 · — | 0.00 · — |
| proc:7 | 1.00 · 37.19 | 0.00 · — | 1.00 · 36.70 | 1.00 · 37.78 | 1.00 · 38.58 | 1.00 · 38.68 |
| proc:1000 | 1.00 · 34.20 | 1.00 · 33.48 | 1.00 · 33.60 | 1.00 · 35.02 | 1.00 · 36.00 | 1.00 · 36.26 |

## Track_A kontrolü

Track_A, yeni pist API'si (`UnityVecEnv(track="Track_A")`: `-trackName`, CONFIG `expected_track_id`) ve aynı süreçte 6 modelle koşuldu. Tüm per-seed metrikleri M5 raporlarıyla (`benchmarks/custom_ppo.json`, `benchmarks/mlagents_bridge.json`) **bit düzeyinde aynı** (6/6).

