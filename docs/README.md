# Otonom Yarış Ajanı: Doküman İndeksi

Unity (C#) ortamı ile Python/PyTorch RL motorunu 5 bağımsız aşamaya (milestone) bölen mimari ve uygulama planı.

**Strateji (Baseline Validation):** Önce ortam ve ödül, hazır ML-Agents PPO ile doğrulanır (M2). Sonra aynı ortam, özel TCP köprüsü (M3) üzerinden sıfırdan yazılmış PyTorch PPO'ya (M4) bağlanır ve benchmark'la kıyaslanır (M5). Böylece fizik ve ödül hataları algoritma hatalarından ayrılır.

## Dosyalar

| Dosya | İçerik |
|---|---|
| [contracts.md](contracts.md) | Ortak sözleşmeler: sürümler, SimConfig, aksiyon ve gözlem şemaları, C# arayüzleri, adım sırası, sonlanma kodları, benchmark protokolü, Env Freeze |
| [milestones/M1_environment_physics.md](milestones/M1_environment_physics.md) | M1: Pist, WheelCollider aracı, checkpoint, sensörler (Unity/C#) |
| [milestones/M2_baseline_reward.md](milestones/M2_baseline_reward.md) | M2: Ödül mühendisliği ve ML-Agents PPO baseline |
| [milestones/M3_bridge_gym.md](milestones/M3_bridge_gym.md) | M3: İkili TCP lockstep köprüsü ve Gymnasium arayüzü |
| [milestones/M4_pytorch_ppo.md](milestones/M4_pytorch_ppo.md) | M4: Sıfırdan PyTorch continuous PPO |
| [milestones/M5_training_benchmark.md](milestones/M5_training_benchmark.md) | M5: Eğitim döngüsü, SPS optimizasyonu, kıyas raporu |

## Bağımlılık grafiği

```
M1 ──► M2 ──► M3 (C#) ──► M5
             ▲           ▲
M3 (Python, mock ile) ───┤
M4 (yalnız contracts.md'ye bağlı, 1. günden paralel) ─┘
```

## Alt ajana görev verme

1. Alt ajana ilgili `milestones/Mx_*.md` dosyası ile `contracts.md` verilir. Başka bağlam gerekmez.
2. Alt ajan DoD maddelerinin her birini kanıtıyla (test çıktısı, log, JSON) raporlar.
3. Sözleşme değişikliği gerekiyorsa önce `contracts.md` güncellenir, sonra etkilenen aşamalar yeniden doğrulanır.
4. M2 DoD'dan sonra **Env Freeze** yürürlüğe girer (C0.11). Ortam, ödül veya gözlem değişikliği baseline'ın yeniden koşturulmasını gerektirir.

## Ortam özeti (2026-09)

- Unity 6000.4.6f1. Donanım: Ryzen 5 5600X, 32 GB RAM, AMD RX 6750 XT (CUDA yok, eğitim yalnız CPU'da).
- Baseline için Python 3.10.11 + `mlagents==1.1.0`. Özel motor için Python 3.13.
- Sistem dili tr-TR. Tüm sayı biçimleme ve ayrıştırma InvariantCulture ile yapılır (C0.13).
