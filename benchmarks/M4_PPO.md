# M4 Doğrulama Raporu: Özel PyTorch PPO Motoru

**Tarih:** 2026-09-23 · **env_config_hash:** `90240ee2b1a58b5b` (C0.14, değişmedi) · **obs_layout_hash:** `b40ca79bdba1c2c2`
**Python:** 3.10.11 (`D:\RaceAgent\.venv`) · **torch 2.14.0+cpu** · numpy 2.2.6 · gymnasium 1.3.0 · tensorboard 2.21.0 · matplotlib 3.10.9 · pytest 9.1.1 · `racing_rl` 0.4.0

## Sonuç

Zorunlu DoD maddelerinin (1–8) hepsi ve ek testler geçti. Opsiyonel DoD 9 (Pendulum) sonucu aşağıdadır. Unity player'la yapılan duman testi de geçti.

| DoD | Ölçüt | Sonuç |
|---|---|---|
| 1 | Şekil ve dtype (M4 tablosu), `evaluate` → [B] | ✅ Tüm tampon şekilleri ve dtype'ları tabloyla aynı. Yanlış şekilde `assert` tetikleniyor. |
| 2 | log π ve H, `torch.distributions.Normal` ile ≤ 1e-6 | ✅ float64'te fark **0.0**. `ActorCritic.evaluate` float32 yolunda log π farkı **4.8e-7**, H farkı 0. |
| 3 | GAE: el hesabı T = 5 (1 terminated + 1 truncated) ve yavaş referans, ≤ 1e-6 | ✅ El hesabı ve referans farkı **0.0** (float64). Buffer yolunda (f64 hesap, f32 saklama) bağıl fark ≤ 1e-6. |
| 4 | Backward, B = 1024 | ✅ Kayıp sonlu (0.279). Aktör, kritik ve `log_std` gradyanlarının hepsi sonlu ve ≠ 0 (en küçük tensör toplamı 1.5e-4). Clip öncesi norm 1.059, sonrası **0.49999961**. `step()` sonrası tüm parametreler değişti. |
| 5 | İlk epoch'un ilk minibatch'inde ρ ≡ 1, `clip_frac = 0`, `approx_kl ≈ 0` | ✅ Gerçek `collect_rollout` sonrası (TargetReach, N=16, T=32) `default` ve `parity` için maxǀρ−1ǀ **0.0**, clip_frac 0, approx_kl 0. Kontrol: obs rollout sonu RMS ile yeniden normalize edilince maxǀρ−1ǀ > 1e-3 (test bunu yakalıyor). |
| 6 | `RunningMeanStd` parçalı ve paralel güncelleme, ≤ 1e-6 bağıl | ✅ 20k × 26 veri, 41 parça (sıralı) ve 7 işçi × 5 parça (Chan birleştirmesi): ortalama ve varyans bağıl farkı **≤ 8.4e-15**. |
| 7 | TargetReach: 30 güncellemede ortalama ödül > −0.05, < 60 s | ✅ Deterministik (μ) ödül **15. güncellemede** −0.039 ile eşiği geçti; 30. güncellemede **−0.020**. Stokastik ödül −0.145 (doğrulanmaz). Test süresi **11.5 s**. |
| 8 | Checkpoint kaydet-yükle: aynı girdi → aynı çıktı; aynı tohum → aynı ilk güncelleme | ✅ Yüklenen modelin deterministik aksiyonu, V(s) değeri, aynı RNG ile stokastik örneği ve RMS'i bit düzeyinde aynı. `torch.load(weights_only=True)` ile okunuyor. Hash uyuşmazlığı `CheckpointMismatchError` veriyor. Aynı tohumla ilk güncellemenin zamanlama dışındaki 27 metriği **bit düzeyinde aynı**: süreç içinde (test) ve iki taze süreç arasında (2 güncelleme, `float.hex` karşılaştırması). |
| 9 | *(opsiyonel, `@slow`)* Pendulum-v1, 1M adım, ortalama dönüş ≥ −250 | ✅ `-m slow`: 245 güncelleme (16 ortam × T 256), **180 s**. 20 taze episode'da deterministik ortalama dönüş **−124.9**. Ayar: lr 1e-3 → 0, γ 0.95, 10 epoch, minibatch 512, c_e 0, `clip_vloss=False`, eğitim ödülü ×0.1 (eval ham ödülle). |
| Ek-a | NaN enjeksiyonu (kayıp NaN / gradyan ∞, 3. minibatch'te) | ✅ Adım atlandı. Model, Adam durumu ve RMS güncelleme başındaki hâline döndü (`torch.equal`). `lr_mult` 0.5 oldu, `RuntimeWarning` verildi, `nan_skipped = 1`. Sonraki güncelleme yarım lr ile sorunsuz koştu. |
| Ek-b | MockUnity uçtan uca | ✅ Gerçek TCP köprüsü üzerinden `train()` 2 güncelleme koştu (4 ajan, truncation yolu dahil). `eval_fn` → `run_eval` (20 ajan) → `per_seed_report` geçerli bir C0.10 sözlüğü üretti. `best.pt` ve `ckpt_*.pt` yazıldı, hash'ler HELLO'dan alındı. `report()` JSON'u serileştirilebilir. |
| Ek-c | Gerçek player (`-m unity`), 16 ajan, 3 güncelleme | ✅ NaN yok. Eğitim **4,446 karar/s** (toplama + güncelleme); yalnız toplama ≈ 5,700 karar/s. Checkpoint yazıldı. Eğitilmemiş politikanın 20 ajanlı eval JSON'u yazıldı. |
| Regresyon | M3 `python/tests` (`not unity`) | ✅ 16/16 |

**pytest özeti:** `pytest -m "not slow and not unity"`: **46 passed** (16 M3 + 30 M4), 7 deselected, 19.3 s. `pytest racing_rl/rl` (varsayılan `-m "not slow"`): 31 passed (unity duman testi dahil; player mevcut), 1 deselected (Pendulum).

## DoD 7 ayrıntısı: öğrenme sağlaması

- **Konfigürasyon (onaylı):** `default` ön ayarı üzerine lr 1e-3, N = 64, T = 64 (güncelleme başına 4096 karar), minibatch 256. Diğer değerler varsayılan: 4 epoch, ε 0.2, c_e 1e-3, γ 0.99, λ 0.95, clip_vloss.
- **Ölçüm:** Her güncellemeden sonra 8192 taze örnekte deterministik μ ile ölçülür (`default_rng(12345)`, RMS güncellenmez).

| Güncelleme | 1 | 5 | 10 | 14 | **15** | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|
| Deterministik ödül | −0.590 | −0.302 | −0.110 | −0.050 | **−0.039** | −0.014 | −0.017 | −0.020 |
| Stokastik ödül | −1.171 | −0.785 | −0.459 | −0.317 | −0.305 | −0.200 | −0.156 | −0.145 |

- **En iyi değer:** Deterministik ödül 22. güncellemede −0.012 ile en iyi noktaya ulaşıyor. Sonrasında −0.02 civarında hafifçe dalgalanıyor; bunun nedeni kritiğin zayıf kalması (not 9).
- **Stokastik ödül:** σ gürültüsü tek başına ≈ 2σ² ceza getiriyor, bu yüzden −0.05 eşiği yalnız deterministik politikaya uygulanıyor (C0.18).
- **Süreler:** Test 30 güncellemenin tamamını koşar ve toplam süreyi 60 s sınırına karşı doğrular. Güncelleme başına ~0.38 s düşer (toplama + 4 epoch × 16 minibatch).
- **Önceki ölçüm:** Taze örnek seti farklıyken (4096 örnek, `default_rng(123)`) eşik 12. güncellemede geçilmişti; 30. güncellemede sonuç −0.023 idi.

## Hiperparametreler

| Parametre | `default` | `parity` |
|---|---|---|
| Aktivasyon / ağ | tanh, 256×2, ayrık aktör-kritik | silu, 256×2 |
| Aksiyon dönüşümü / `log_std` başlangıcı | `clip` / −0.5 (`clamp[−5, 1]`) | `mlagents_scale` (clip(u,−3,3)/3) / 0 |
| lr | 3e-4 → 0 (lineer), Adam eps 1e-5 | 3e-4 → 1e-10 (lineer) |
| ε (politika kırpma) | 0.2 sabit | 0.2 → 0.1 |
| c_e (β) | 1e-3 sabit | 5e-3 → 1e-5 |
| c_v / ε_v / `clip_vloss` | 0.5 / 0.2 / True | 0.5 / 0.2 / True |
| γ / λ | 0.99 / 0.95 | 0.99 / 0.95 |
| epoch / minibatch / T | 4 / 1024 / 256 | 3 / 1024 / 20480 // N_total (N=32 → 640) |
| `norm_obs` / `obs_clip` / ‖∇‖ | True / 10 / 0.5 | True / 10 / 0.5 |
| `target_kl` | yok | yok |
| torch iş parçacığı (toplama / güncelleme) | 2 / 4 | 2 / 4 |

## Dosya düzeni (`python/racing_rl/rl/`)

| Dosya | İçerik |
|---|---|
| `config.py` | `PPOConfig` (frozen dataclass), `presets` (`default`, `parity`), `get_preset`, `load_yaml`, `to_dict`/`from_dict`, `LinearSchedule` |
| `normalization.py` | `RunningMeanStd` (float64, Chan paralel birleştirme, `normalize(clip=10)`, eps 1e-8) |
| `distributions.py` | Köşegen Gauss `log_prob`/`entropy`/`sample`, `clip` ve `mlagents_scale` dönüşümleri |
| `networks.py` | `ActorCritic`: `act`, `act_full`, `act_deterministic`, `evaluate`, `value`; `obs_rms` modülün üzerinde |
| `buffer.py` | `RolloutBuffer` (önceden ayrılmış f32 tensörler, `compute_gae`, `minibatches`) |
| `gae.py` | Vektörize `compute_gae` ve testler için yavaş `gae_reference` |
| `ppo.py` | `PPOTrainer` (`update`, `compute_loss`, NaN koruması, annealing) |
| `policy.py` | `TorchPolicy`, `save_checkpoint`/`load_checkpoint`, `actor_critic_from_checkpoint`, `restore_rng` |
| `synthetic_env.py` | `TargetReachVecEnv` (SAME_STEP, `final_obs`, truncation) |
| `rollout.py` | `collect_rollout`, `EpisodeStats` (kapsam eklemesi) |
| `loop.py` | `train()` (checkpoint rotasyonu, `eval_fn` → `best.pt`; kapsam eklemesi) |
| `tests/` | `test_shapes`, `test_distributions`, `test_gae`, `test_backward`, `test_ppo_invariant`, `test_normalization`, `test_learning`, `test_checkpoint`, `test_nan_guard`, `test_mock_integration`, `test_unity_smoke` (`unity`), `test_pendulum` (`slow`) |

## Uygulama notları ve sapmalar

1. **Başlatma determinizmi (bulgu):** `nn.init.orthogonal_`'ın kullandığı LAPACK QR, `torch.set_num_threads` değerine bağlı olarak ~1e-6 farklı ağırlık üretiyor. Aynı süreçte iki koşu arasında (güncelleme 4 iş parçacığı bırakıyor) ilk güncelleme metrikleri bu yüzden 1e-9 düzeyinde ayrışıyordu. `ActorCritic` artık başlatmayı tek iş parçacığında yapıyor ve aynı tohum her durumda aynı ağırlıkları veriyor.
2. **Protokol genişlemeleri:** `act()` sözleşmedeki dörtlüyü döndürür. Buffer'a yazılacak normalize gözlem için ayrıca `act_full()` (beşli, `obs_n` dahil) ve `last_obs_n` alanı var. M3 köprüsünün `Policy` protokolü `__call__` olduğundan `TorchPolicy` hem `__call__` hem `act(obs, deterministic)` uygular.
3. **NaN koruması:** "Son checkpoint yüklenir" kuralı bellek içi anlık görüntüyle uygulanıyor (model + optimizer + RMS; diske gitmez, her güncellemede taze). `lr_mult` kalıcı ve checkpoint'e (`extra.trainer`) yazılıyor.
4. **Sonlu olmayan gözlem:** RMS'i kalıcı bozmasın diye 0'a çevriliyor ve uyarı veriliyor (M3 `UnityVecEnv` de uyarıyor).
5. **`collect_rollout` geri dönüşü:** `step_async`/`step_wait` kullanılıyor. Yalnızca bunlar olmayan ortamlarda (Pendulum testindeki gymnasium `SyncVectorEnv`) `step()` kullanılıyor.
6. **Minibatch:** T·N minibatch boyutuna bölünmüyorsa son parça da kullanılıyor; yalnız < 2 örnekli parça atlanıyor (std tanımsız).
7. **`explained_var`:** Rollout anındaki V ile R üzerinden hesaplanıyor (CleanRL/SB3 ile aynı).
8. **Kurulum yan etkisi:** torch kurulumu `setuptools`'u 65.5.0 → 78.1.0 yükseltti. Başka paket değişmedi.
9. **M5 için not:** TargetReach'te `explained_var` negatif kalıyor. Bu beklenen bir durum: hedef her adımda yeniden örneklendiği için gelecek ödüller eylemden bağımsız gürültü, ε_v = 0.2 değer kırpması da kritiğin ~−100 ölçeğindeki dönüşlere hızla yaklaşmasını sınırlıyor. Politika buna rağmen öğreniyor (avantaj normalizasyonu). Unity duman testinde `explained_var` 0.14–0.60 arasında.

## Yeniden üretme

```
cd D:\RaceAgent\python
..\.venv\Scripts\python.exe -m pytest -m "not slow and not unity" --durations=10   # 46 test
..\.venv\Scripts\python.exe -m pytest racing_rl/rl -m unity -s                      # player duman testi (port 6405/6435)
..\.venv\Scripts\python.exe -m pytest racing_rl/rl -m slow -s                       # DoD 9, Pendulum 1M
```
