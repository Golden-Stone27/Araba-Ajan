## Özet

- **DoD geçti.** Özel PyTorch PPO'nun `best.pt` modelleri test tohumlarında (1000..1019) **39.18 s** tohum medyanı verdi. Bu, T_ref'in (41.22 s) **%4.95 altında**. Üç tohumun hepsi ML-Agents'ı geçti: 39.18 / 38.68 / 40.50 s. Her tohumda completion 1.00.
- **ML-Agents köprü değerlendirmesi M2/M3 ile birebir aynı:** 41.28 / 41.22 / 41.08 s. Resmi kıyasta iki politika da aynı build, aynı değerlendirici (`racing_rl.train.evaluate`) ve aynı tohumlarla koşuldu.
- **Hız farkının kaynağı düzlüklerdeki son hız.**
  - Özel PPO düzlüklerde tam gaza gidiyor ve 134–140 km/h'e çıkıyor; ML-Agents 115–125 km/h'te kalıyor. Tam gazda geçen süre payı 0.36–0.40'a karşı 0.06–0.29.
  - Fren noktaları ve viraj içi en düşük hızlar neredeyse aynı (≈ 50 km/h).
  - Kazanç sektör 1 ve 3'te (düzlükler): tohum medyanı −0.93 s ve −0.77 s. Sektör 2 (virajlı bölüm) +0.21 s ile ML-Agents lehine.
- **Yarış çizgisi farklı.** Özel PPO düzlüklerde orta çizginin 2–3.5 m soluna yerleşiyor, ML-Agents 1–3 m sağında (`e_lat` grafiği). İki politika virajlara farklı taraftan giriyor.
- **Bedeli direksiyon pürüzsüzlüğü.** `steer_smoothness` 0.0358'e karşı 0.0197 (+%82). Özel PPO virajlarda daha sert ve titrek direksiyon kullanıyor; 2. tohumda sonuna kadar kırıp (a0 = −1) düzeltme yapıyor. Ödüldeki `wSmooth` cezası (0.02) hızın getirisine göre küçük kaldığı için ajan bu takası kabul ediyor.

## Zirve sonrası gerileme ve model seçimi (C0.19)

Özel PPO'da doğrulama tur süresi 1.5–4M karar arasında en iyi değerine ulaşıyor. Sonra tohuma göre 1.5–4 s geriliyor ve lr sıfıra yaklaştıkça kısmen toparlanıyor. Tohum başına en iyi nokta ve 10M'deki son değer:

| Tohum | En iyi (adım) | 10M son |
|---|---|---|
| 1 | 39.18 s (2.05M) | 42.80 s |
| 2 | 38.68 s (4.10M) | 40.70 s |
| 3 | 40.50 s (1.54M) | 42.66 s |

- Gerileme dönemlerinde eğitim ödülü de düşüyor (s1: 209 → 197), yani bu yalnız "ödülü artırıp turu yavaşlatma" değil.
- Aynı dönemlerde stokastik politikada ara ara duvar çarpması patlamaları görülüyor (%12–25). Ardından politika temkinli sürüşe kayıyor ve σ küçüldükçe bu noktadan tam çıkamıyor.
- ML-Agents s1'de böyle bir gerileme yok (1M: 43.31 s → 10M: 41.28 s). Olası nedenler C0.19'daki kod düzeyindeki farklar (minibatch avantaj normalizasyonu, T = 160 GAE ufku, gradyan kırpma) ile 128 paralel ajanla güncelleme başına daha çeşitli veri. Bu M5'te ayrıca izole edilmedi.
- **Model seçimi:** Kullanıcı kararıyla birincil metrik `best.pt`. Seçim doğrulama tohumlarında (2000..2019) yapıldı, test tohumlarında (1000..1019) raporlandı. EvalGrid pertürbasyonu (±0.5 m, ±2°) deterministik politika için çok küçük olduğundan doğrulama ve test sonuçları tohum başına aynı çıktı (ör. s1 39.18 / 39.18 s). Yani seçim yanlılığı pratikte ölçülemeyecek kadar küçük. Son checkpoint ve 1M anlık görüntüsü aşağıdaki tablolarda ayrı sütunlarda veriliyor.
- **Son checkpoint ile kıyas:** T = 42.66 s, T_ref'e göre +%3.5; DoD'u geçmiyor.
- **1M anlık görüntüsü ile kıyas:** T = 43.10 s. ML-Agents'ın 1M modelleri (s2 41.22 s, s3 41.08 s) aynı adımda daha iyi. Özel PPO'nun zirvesi daha geç geliyor (1.5–4M).
- **Örnek verimliliği:** Eğitim telemetrisinde ilk 3 tur 799k kararda (ML-Agents: 440k), %95 completion 1.11M kararda (ML-Agents: 680k). Buna karşın duvar saati verimliliği çok daha yüksek (aşağıda).

## Hız (SPS) ve altyapı

| | ML-Agents (M2) | Özel PPO (M5) |
|---|---|---|
| Paralellik | 3 süreç × 16 ajan, `time_scale 20`, gRPC | 4 süreç × 32 ajan, script modu, özel TCP köprüsü |
| Eğitim hızı (toplama + güncelleme) | ~3,000 karar/s (tek koşu) | **~14,000 karar/s** (toplama ~22k, güncelleme 0.5 s / 20,480 örnek) |
| 1e7 karar | 0.94 sa | **0.23 sa** (13.8–15.3 dk, eval dahil) |

- **SPS probu** (`eval/m5_sps_probe.json`, `eval/m5_sps_probe_policy.json`):
  - Rastgele aksiyonla ölçekleme doğrusala yakın: 8×1 = 6.5k, 16×4 = 25.5k, 32×4 = 28.9k karar/s.
  - p99 gecikme en fazla 5.8 ms (sınır 50 ms).
  - Politika çalışırken 32×4 = 23.7k karar/s.
- **Tekrar üretilebilirlik** (`eval/m5_repro.json`): iki taze süreçte (4 Unity + Python) aynı config ve tohumla ilk 2 güncellemenin 42 metriği `float.hex` düzeyinde aynı.
- **Hata yönetimi:**
  - Üç koşuda da 0 Unity restart.
  - Çökme kurtarma, `--resume` ve Ctrl+C yolları mock testlerle doğrulandı (`tests/test_m5_train.py`). Kapsanan senaryolar: süreç ölümü → rollout atılır → yeniden başlatma; kaldığı yerden devam; TensorBoard ve CSV satırlarının temizlenmesi.

## Parite ve sapmalar

- **Kayıp ölçeği (config ile):**
  - ML-Agents entropinin aksiyon boyutları üzerinden ortalamasını, `racing_rl` toplamını alıyor; bu yüzden `ent_coef` 2.5e-3 → 5e-6 (β/2) yapıldı.
  - Değer kaybı için `vf_coef` 1.0 kullanıldı.
  - Bu ayarlarla kayıp terimleri ML-Agents 1.1 ile birebir aynı (C0.19).
- **Kod düzeyindeki farklar (değiştirilmedi, M4 mühürlü):**
  - Avantaj normalizasyonu minibatch bazında (ML-Agents: tüm buffer).
  - GAE ufku T = 160 (ML-Agents `time_horizon` 128).
  - Gradyan normu 0.5 ile kırpılıyor (ML-Agents kırpmıyor).
  - ε_v sabit 0.2.
- **Kapsam:** Ayar merdiveni kullanılmadı; `parity` ön ayarı (1. basamak) hedefi ilk denemede geçti. `ppo_default.yaml` ablasyonu koşulmadı.
