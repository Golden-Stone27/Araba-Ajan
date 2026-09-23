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
