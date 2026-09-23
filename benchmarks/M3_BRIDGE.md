# M3 Doğrulama Raporu: Özel TCP Köprüsü

**Tarih:** 2026-09-23 · **env_config_hash:** `90240ee2b1a58b5b` (C0.14, değişmedi) · **obs_layout_hash:** `b40ca79bdba1c2c2`
**Build:** `Builds/RaceEnv/RaceEnv.exe` (Unity 6000.4.6f1, sahne `Race_Bridge.unity`) · **Python:** 3.10.11, numpy 2.2.6, gymnasium 1.3.0, onnxruntime 1.23.2

## Sonuç

Tüm DoD maddeleri geçti. Köprü üzerinden ONNX ile sürülen M2 modelleri, M2'deki in-Unity (ML-Agents) sonuçlarını **tur tur** yeniden üretti.

| DoD | Ölçüt | Sonuç |
|---|---|---|
| 1 | `pytest -m "not unity"` (protokol + mock senaryoları) | ✅ 16/16 |
| 2 | `check_env(RacingEnv(exe, num_agents=1))` | ✅ uyarısız geçti |
| 3 | Kayıpsızlık: N=16, 100k STEP (1.6M ajan-kararı) | ✅ desync 0, zaman aşımı 0. Python 100,001 istek / 100,001 yanıt; Unity 100,003 alınan / 100,003 gönderilen (el sıkışma ve CLOSE dahil) |
| 4 | Determinizm: iki taze süreç, aynı tohum, `default_rng(7)` aksiyonları, 5000 karar | ✅ obs, ödül ve bayraklar `np.array_equal`; 863 oto-reset; akış SHA-256'ları aynı |
| 5 | Ödül tutarlılığı: \|Σ reward − `ep_return`\| ≤ 1e-4 | ✅ 17,507 episode, en büyük fark 2.5e-7 |
| 6 | Parite: completion ±5 puan, flying lap medyanı ±%1 | ✅ 3 tohumda da Δcompletion 0, Δmedyan 0.00% |
| 7 | Performans: ≥ 2000 ajan-kararı/s; ek yük p50 < 0.3 ms, p99 < 2 ms | ✅ **7,609 karar/s**; ek yük p50 **0.054 ms**, p99 **0.090 ms** |
| — | Sıfır GC (kararlı STEP işleme, Unity) | ✅ 99,900 STEP'te 0 bayt |

## Parite (DoD 6): köprü + onnxruntime ve ML-Agents in-proc karşılaştırması

C0.10 protokolü: 20 ajan, EvalGrid (tohum 1000+i), 3 tur, deterministik μ. ONNX giriş ve çıkışı `onnx.load` ile doğrulandı: `obs_0` → `deterministic_continuous_actions`.

| Tohum | completion (köprü / M2) | Flying medyan (s) | Flying en iyi (s) | 1. tur (s) | steer_smoothness | 40 flying lap |
|---|---|---|---|---|---|---|
| 1 | 1.00 / 1.00 | 41.28 / 41.28 | 41.28 / 41.28 | 44.32 / 44.32 | 0.01818 / 0.01818 | **40/40 aynı** |
| 2 | 1.00 / 1.00 | 41.22 / 41.22 | 41.20 / 41.20 | 44.40 / 44.40 | 0.01971 / 0.01971 | 39/40 aynı, 1 tur −0.02 s (tek fizik adımı) |
| 3 | 1.00 / 1.00 | 41.08 / 41.08 | 41.06 / 41.06 | 44.27 / 44.27 | 0.02086 / 0.02086 | **40/40 aynı** |

- **T_ref (köprü) = 41.22 s**, M2'de tescil edilen değerle aynı.
- **s2'deki tek fark:** onnxruntime ile Unity Inference Engine (Burst) arasındaki float farkından geliyor. Bu fark bir turu tek fizik adımı kaydırmış.
- **Ortalama hız:** Köprüde 26.62 / 26.76 / 26.73 m/s, M2'de 26.62 / 26.75 / 26.73. Sektör süreleri de ±0.02 s içinde. Bu metrikleri köprü 0.1 s'lik karar örneklemesiyle hesaplıyor (C0.17), in-proc değerlendirici 0.02 s'lik fizik adımıyla.
- **Süre:** Değerlendirme tohum başına ~4 s sürüyor. M2'de in-proc değerlendirme `time_scale 20` ile çalışıyordu.

## Performans (DoD 7), tek süreç, N=16, rastgele aksiyon, `-batchmode -nographics`

| | p50 | p99 | ortalama |
|---|---|---|---|
| Python RTT (µs) | 1988 | 2627 | 2021 |
| Unity STEP işleme (K=5 fizik adımı + gözlem + STATE yazımı, µs) | 1932 | 2557 | 1965 |
| **Köprü ek yükü = RTT − Unity (µs)** | **54** | **90** | p90 66, en kötü tekil örnek 3047 |

- Hız 475.6 STEP/s = **7,609 ajan-kararı/s** (hedef ≥ 2000). Süreyi neredeyse tamamen fizik belirliyor (%97).
- 100k örneğin tamamı `seq` üzerinden eşlendi (`-bridgeTimingLog`).
- STATE payload'u N=16 için 4064 B.
- `MultiUnityVecEnv` (2 süreç × 16 ajan, duman testi): **13,895 ajan-kararı/s**. Portlar 6005/6030.

## Uygulamada bulunan ve çözülen sorunlar

1. **Reset tarihçe bağımlılığı (M1 bulgusu):**
   - **Sorun:** `VehicleController.TeleportTo`, WheelCollider'ın iç PhysX durumunu (süspansiyon, temas, lastik) sıfırlamıyor. Aynı tohumla yapılan iki reset farklı yörünge üretiyordu: gözlem farkı ~0.5, ayrıca `grounded` ilk reset'te 0, sonrakilerde 1. `check_env` bu yüzden başarısız oluyordu.
   - **Denenen:** `TeleportTo` içinde tekerleri kapatıp açmak. Bu, ilk episode dinamiğini de değiştirdi (s1 flying 41.28 → 41.12 s). Env Freeze'i bozacağı için **geri alındı**.
   - **Uygulanan:** Köprü, açık `RESET` geldiğinde araçları yeniden kuruyor (`RaceEnvironment.RebuildAgents`, yalnızca ekleme; Core adım mantığı aynı). Bunun sonuçları:
     - `RESET(seed)` aynı süreç içinde bit düzeyinde tekrarlanabilir.
     - Başlangıç koşulu M2 eval'iyle birebir aynı (parite tablosu).
     - Oto-reset'ler M2 eğitimiyle aynı davranıyor.
   - **Not (M5):** Eğitimdeki oto-reset'ler tarihçeye bağlı kalıyor. Bu yalnızca bir gürültü kaynağı; süreçler arası determinizm (DoD 4) bundan etkilenmiyor.
2. **`ep_return` float32 kayması:** 15k adımlık birikimde ~1e-3'e ulaşıyordu. `RaceAgentCore` içinde double ile biriktiriliyor. Yalnız telemetriyi etkiliyor; ödüller ve hash değişmedi.
3. **Blok içi reset ödülü:** Ajan blok ortasında sıfırlandığında, yeni episode'unda blok sonuna kadar kazandığı ödül bir sonraki STATE'e aktarılıyor. Aksi hâlde Σ reward ile `ep_return` tutmazdı.
4. **Doküman hatası:** M3 dokümanındaki "N=16 için 4448 B" değeri yanlıştı; alan alan yerleşimden hesaplanan doğru değer 4064 B. Doküman düzeltildi.

## Testler

- **Unity EditMode:** 33/33 (yeni: `BridgeProtocolTests` ×3; donmuş hash testi geçti).
- **Unity PlayMode:** 8/8 (M1 referansları değişmedi: PurePursuit 54.58 / 51.78 s, 0–100 km/h 6.04 s).
- **pytest:** 21/21 (16 mock/protokol + 5 `-m unity`).

## Dosyalar

- `eval/m3_bridge_validation.json`: tüm DoD ölçümleri (ham).
- `eval/bridge_s{1,2,3}.json`: `race-benchmark/v1`, `evaluator: "bridge"`.
- **Yeniden üretme:** `python python/scripts/m3_validate.py` (tamamı ~5 dk) veya `python -m racing_rl.bridge.evaluate --exe ... --model ... --train-seed N --out ...`.
