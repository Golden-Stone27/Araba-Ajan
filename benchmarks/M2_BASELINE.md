# M2 Baseline Raporu: ML-Agents PPO

**Tarih:** 2026-09-23 · **env_config_hash:** `90240ee2b1a58b5b` (Env Freeze, contracts.md C0.14) · **obs_layout_hash:** `b40ca79bdba1c2c2`
**Unity:** 6000.4.6f1 · `com.unity.ml-agents` 4.1.0 · Python 3.10.11, `mlagents` 1.1.0, torch 2.2.2+cpu (yalnız CPU)

## Sonuç

| Tohum | Model (eğitim adımı) | completion_rate | Flying lap medyan / en iyi | 1. tur | Ort. hız | Direksiyon düzgünlüğü (steer_smoothness) | Sektörler (s) |
|---|---|---|---|---|---|---|---|
| 1 | 10,000,031 | **1.00** (20/20) | 41.28 / 41.28 s | 44.32 s | 26.6 m/s | 0.018 | 13.18 / 14.86 / 13.26 |
| 2 | 999,947 | **1.00** (20/20) | 41.22 / 41.20 s | 44.40 s | 26.8 m/s | 0.020 | 12.80 / 15.24 / 13.18 |
| 3 | 999,950 | **1.00** (20/20) | 41.08 / 41.06 s | 44.27 s | 26.7 m/s | 0.021 | 12.68 / 15.20 / 13.20 |

**T_ref = 41.22 s** (tohum medyanı). Referans: PurePursuit otopilotu 51.78 s. Seed 1'in 1M checkpoint'i 43.31 s idi (`eval/mlagents_s1_1M.json`); 1e7'ye uzatmak turu ~2 s kısalttı ve `steer_smoothness`'ı 0.037'den 0.018'e indirdi.

Eval protokolü C0.10: 20 ajan, EvalGrid (s = 1 m, tohum 1000 + i), 3 tur, deterministik (μ) çıkarım, `evaluator: unity-inproc`.

## Eğitim

| Tohum | İlk 3 tur (karar) | Eğitimde %95 completion (karar) | Süre |
|---|---|---|---|
| 1 | 400k | 680k | 0.94 sa (1e7) |
| 2 | 440k | 820k | 0.11 sa (1M) |
| 3 | 500k | 680k | 0.11 sa (1M) |

- 3 süreç × 16 ajan, `time_scale 20`. Tek koşu ~3,000 karar/s; iki koşu paralelken ~2,400 karar/s.
- Öğrenme sırası (seed 1): 0–200k çoğu episode Stuck (%98) → 200–600k hızlanıyor ama duvara çarpıyor (Wall %70–90) → 700k+ duvar kaçınma. 900k'dan itibaren tüm episode'lar 3000 kararlık TimeLimit'e ulaşıyor.
- Seed 1, 1e7 adımda: ödül 209, episode başına ~6.1 tur, entropi 1.42 → 0.30.
- Tohum 2 ve 3 aynı `race_ppo.yaml` ile başlatıldı. ~1M checkpoint export edildikten sonra durduruldular; lr/β/ε planları seed 1'in 1M anıyla aynı.

## Dosyalar

- `baseline_mlagents.json`: C0.10 şeması. Tohum başına eğitim eğrisi (100k aralıklı) ve checkpoint bilgisi içerir.
- `models/mlagents_baseline_s{1,2,3}.onnx`: değerlendirilen modeller.
- `eval/*.json`: `BenchmarkRunner` ham çıktıları.
- Yeniden üretme: `mlagents/make_baseline_report.py` (kullanımı dosya başında).
