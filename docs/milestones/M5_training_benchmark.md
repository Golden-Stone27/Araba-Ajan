# M5: Eğitim Döngüsü, Optimizasyon ve Kıyaslama

> **Alt ajan talimatı:** Bu aşamayı uygulamak için yalnızca bu dosya ve [`../contracts.md`](../contracts.md) yeterlidir.
> Sözleşmelerdeki (C0.x) bir değeri değiştirmen gerekirse, değiştirmeden önce gerekçesiyle raporla.
> **Bağımlılıklar:** M2 (benchmark + ONNX), M3 (köprü), M4 (PPO).
> **Teslim edilen (sonraki aşamaya):** Nihai: `custom_ppo.json`, `REPORT.md`, grafikler.
> **Kural:** DoD'daki tüm maddeler geçmeden aşama tamamlanmış sayılmaz.

**Dosyalar:**
- `racing_rl/train/{train_ppo.py, evaluate.py, telemetry.py, compare.py, sps_probe.py}`
- `configs/{ppo_parity.yaml, ppo_default.yaml}`
- `benchmarks/{custom_ppo.json, mlagents_bridge.json, REPORT.md, plots/}`

**Döngü:**

```
envs = MultiUnityVecEnv(exe, procs=K, agents=N, expected_env_hash=FROZEN)  ; obs,_ = envs.reset(seed)
for update in range(U):
  progress = global_step / total_steps ; annealing (lr, ε, c_e)
  for t in range(T):
    with no_grad: a_env, u, logp, v = ac.act(obs, update_rms=True)     # torch threads = 1–2
    envs.step_async(a_env); next_obs, r, term, trunc, info = envs.step_wait()
    if trunc.any(): r[trunc] += γ · ac.value(info["final_obs"][trunc])
    buf.add(obs_n, u, logp, v, r, term | trunc) ; telemetry.on_step(info) ; obs = next_obs
  buf.compute_gae(ac.value(obs)) ; m = trainer.update(buf, progress)   # torch threads = 4
  log(m); her 10 güncellemede checkpoint; her 25 güncellemede eval (ayrı süreç, N = 20, C0.10) → best.pt
```

- **Zaman ölçeği:** Script modunda simülasyon CPU'nun izin verdiği hızda koşar. Δt sabit olduğu için dinamik M2 ile aynıdır.
- **Replay:** PPO on-policy çalıştığı için yalnızca rollout buffer kullanılır. SAC kapsam dışıdır; `Policy` ve `VectorEnv` sözleşmeleri SAC eklenmesine açıktır.

**SPS optimizasyon prosedürü:**
1. `sps_probe.py` ile rastgele aksiyonla ortamın üst sınırı ölçülür. N ∈ {8, 16, 32} ve K ∈ {1, 2, 3, 4} tablosu çıkarılır.
2. p99 step gecikmesi < 50 ms olan en yüksek SPS konfigürasyonu seçilir.
3. Numpy işlemleri vektörize edilir; ajan başına Python döngüsü kullanılmaz.
4. Gerekirse Unity 6 Dedicated Server build alt hedefi denenir.
5. Hedef: 5600X üzerinde toplam ≥ 3000 karar/s. Bu hızda 1e7 karar yaklaşık 1 saat sürer.

**Telemetri (TensorBoard ve CSV):**
- x ekseni ortam kararıdır (ML-Agents "Step" ile aynı).
- ML-Agents ile aynı etiket adları kullanılır, böylece eğriler üst üste çizilebilir:
  - `Environment/Cumulative Reward`, `Environment/Episode Length`
  - `Losses/Policy Loss`, `Losses/Value Loss`
  - `Policy/Entropy`, `Policy/Learning Rate`, `Policy/Epsilon`, `Policy/Beta`
- Ek etiketler:
  - `Race/LapTime`, `Race/CompletionRate`, `Race/TermReason/<ad>`
  - `Train/ApproxKL`, `Train/ClipFrac`, `Train/ExplainedVar`, `Train/GradNorm`, `Train/Sigma_{steer,thr}`
  - `Perf/SPS`, `Perf/StepLatencyP50|P99`
- Her koşu için `config.yaml`, `git_sha`, `env_config_hash` ve tohum kaydedilir.

**Değerlendirme ve kıyas:**
- `evaluate.py --policy torch:<ckpt>|onnx:<path> --exe ... --out <json>`. C0.10 protokolünü **köprü üzerinden** uygular. Resmi kıyas iki politika için de aynı build ve aynı değerlendiriciyle yapılır.
- M2'nin in-Unity sonucu yalnızca çapraz kontrol içindir (±%1).
- `compare.py`:
  - `env_config_hash` eşitliği zorunludur.
  - Çıktı tablosu: metrik | ML-Agents PPO | Özel PPO | Δ%
  - Grafikler: öğrenme eğrisi (ödül ve tur süresinin karara göre değişimi), tur süresi kutu grafiği, **yarış çizgisi bindirmesi** (info içindeki `pos_x`, `pos_z`), **hız ve s profili**, **sektör süresi farkları**
  - Örnek verimliliği: ilk 3 tura ve %95 tamamlama oranına ulaşana kadar geçen karar sayısı; duvar saati süresi
  - Sonuç `REPORT.md` dosyasına yazılır.

**Hata yakalama:**
- Unity çökerse veya zaman aşımı olursa ilgili süreç yeniden başlatılır, mevcut rollout atılır ve eğitim sürer (en fazla 3 kez/saat, aşılırsa durulur).
- NaN kaybında M4 kuralı uygulanır.
- Ctrl+C ile checkpoint kaydedilir ve tüm Unity süreçleri `kill_tree` ile kapatılır.
- `--resume` ile kaldığı yerden devam edilir.
- Disk: en fazla 10 checkpoint tutulur (ML-Agents ile aynı).

**Hedef tutturulamazsa ayar merdiveni** (yalnız algoritma tarafı; ortam dondurulmuştur):
1. `parity` ön ayarı
2. Adım sayısını 2 katına çıkar
3. γ = 0.995 (etkin ufuk 10 s'den 20 s'ye çıkar)
4. Dönüş std'sine göre ödül normalizasyonu
5. lr 1e-4 veya KL-adaptif lr
6. 512×2 ağ veya duruma bağlı σ
7. c_e ayarı

**Teşhis kuralları:**
- `ExplainedVar < 0`: kritik veya ödül ölçeği sorunu
- `approx_kl > 0.05`: lr çok yüksek
- σ sürekli büyüyor: avantaj işareti ya da log π hatası (M4 testi 5'e dön)
- Erken entropi çöküşü: c_e ↑

**DoD:**
- 3 tohum için özel PPO'nun (köprü eval) flying lap medyanları alınır. Bunların medyanı **≤ T_ref × 1.01** olmalı.
- Her tohumda `completion_rate ≥ 0.95`.
- `env_config_hash` eşit.
- `REPORT.md` ve grafikler üretilmiş.
- Koşular yeniden üretilebilir: aynı config ve tohumla ilk güncelleme metrikleri aynı çıkar.
