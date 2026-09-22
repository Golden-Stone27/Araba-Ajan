# M4: Özel PyTorch PPO Motoru (Python)

> **Alt ajan talimatı:** Bu aşamayı uygulamak için yalnızca bu dosya ve [`../contracts.md`](../contracts.md) yeterlidir.
> Sözleşmelerdeki (C0.x) bir değeri değiştirmen gerekirse, değiştirmeden önce gerekçesiyle raporla.
> **Bağımlılıklar:** Yalnızca `contracts.md` (ilk günden paralel yürütülebilir).
> **Teslim edilen (sonraki aşamaya):** M5: `racing_rl.rl` paketi (`ActorCritic`, `PPOTrainer`, `RolloutBuffer`, `TorchPolicy`).
> **Kural:** DoD'daki tüm maddeler geçmeden aşama tamamlanmış sayılmaz.

**Dosyalar (`racing_rl/rl/`):**
- `config.py` (`PPOConfig` dataclass ve YAML yükleyici)
- `normalization.py` (`RunningMeanStd`, paralel Welford)
- `networks.py`
- `distributions.py`
- `buffer.py` (`RolloutBuffer`)
- `gae.py`
- `ppo.py` (`PPOTrainer`)
- `policy.py` (`TorchPolicy`, checkpoint kaydetme ve yükleme)
- `synthetic_env.py` (`TargetReachVecEnv`, aynı VectorEnv sözleşmesi, SAME_STEP)
- `tests/test_*.py`

**Mimari:**
- Ayrık aktör ve kritik (paylaşımlı gövde yok).
- **Aktör:** O(26) → 256 → 256 → A(2) çıkışı μ. `log_std`, duruma bağlı olmayan bir `nn.Parameter(A)`. Başlangıç −0.5, `clamp[−5, 1]`.
- **Kritik:** 26 → 256 → 256 → 1.
- Aktivasyon `tanh` (varsayılan) veya `silu` (parite ön ayarı).
- Başlatma: ortogonal. Gizli katmanlarda gain √2, μ başında 0.01, V başında 1.0. Bias 0.
- **Aksiyon dönüşümü:**
  - `clip` (varsayılan): u ~ N(μ, σ²), ortama giden `a = clip(u, −1, 1)`. log π, u üzerinden hesaplanır. Buffer'da u saklanır.
  - `mlagents_scale`: `a = clip(u, −3, 3) / 3`, `log_std` başlangıcı 0. ML-Agents ile birebir eşleşir.
- Deterministik eylem `a = transform(μ)`.

**Matematik:**

```
log π(u|s) = Σ_j [ −(u_j−μ_j)²/(2σ_j²) − log σ_j − ½ log 2π ]
H          = Σ_j [ ½ + ½ log 2π + log σ_j ]
Truncation bootstrap:   truncated_t ise r_t ← r_t + γ · V(final_obs_t)
δ_t  = r_t + γ (1 − d_t) V(s_{t+1}) − V(s_t)                 d_t = terminated_t ∨ truncated_t
Â_t  = δ_t + γλ (1 − d_t) Â_{t+1} ;   Â_{T} başlangıcı: V(s_T) = last_values
R_t  = Â_t + V(s_t)
ρ_t  = exp(log π_θ(u_t|s_t) − log π_old(u_t|s_t))
Â    ← (Â − mean_mb) / (std_mb + 1e−8)                        (minibatch başına)
L_π  = −E[ min(ρ Â, clip(ρ, 1−ε, 1+ε) Â) ]
L_V  = ½ E[ max((V_θ−R)², (V_old + clip(V_θ−V_old, −ε_v, ε_v) − R)²) ]   (clip_vloss=True)
L    = L_π + c_v L_V − c_e H ;   ‖∇‖₂ ≤ 0.5 (clip_grad_norm_)
approx_kl = E[(ρ − 1) − log ρ] ;  target_kl aşılırsa epoch erken kesilir (opsiyonel)
explained_var = 1 − Var(R − V)/Var(R)
```

**Tensör şekilleri** (T = rollout uzunluğu, N = toplam ajan):

| Tampon | Şekil | dtype |
|---|---|---|
| `obs` (toplama anındaki RMS ile normalize edilmiş) | [T, N, 26] | f32 |
| `u` (kırpılmamış) | [T, N, 2] | f32 |
| `logp`, `values`, `rewards`, `dones` | [T, N] | f32 |
| `last_values` | [N] | f32 |
| `adv`, `returns` | [T, N] | f32 |

Minibatch öncesinde tamponlar [T·N, …] boyutuna düzleştirilir ve `randperm` ile karıştırılır.

**`PPOConfig` ön ayarları:**
- **`parity`** (ML-Agents ile birebir):
  - lr 3e-4, lineer azalarak 1e-10'a
  - ε 0.2, lineer azalarak 0.1'e
  - c_e 5e-3, lineer azalarak 1e-5'e
  - γ 0.99, λ 0.95, epoch 3, minibatch 1024
  - rollout = 20480 / N_total adım (örneğin 32 ajan için 640)
  - `silu`, 256×2, `norm_obs = True`, `mlagents_scale`
- **`default`**:
  - `tanh`, `clip`, `log_std` başlangıcı −0.5, epoch 4, rollout 256, minibatch 1024, c_v 0.5
  - Adam eps 1e-5, `obs_clip` 10, `torch_threads` 2
- Protokoller:

```python
class Policy(Protocol):
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray: ...
class ActorCritic(nn.Module):
    def act(self, obs_raw: np.ndarray, update_rms: bool) -> tuple[np.ndarray, Tensor, Tensor, Tensor]  # a_env, u, logp, v
    def evaluate(self, obs_n: Tensor, u: Tensor) -> tuple[Tensor, Tensor, Tensor]                        # logp, H, v
    def value(self, obs_raw: np.ndarray) -> Tensor
class PPOTrainer:
    def update(self, buf: RolloutBuffer, progress: float) -> dict[str, float]  # progress∈[0,1]: annealing
```

- **Checkpoint içeriği:** `{model, optimizer, obs_rms{mean,var,count}, config, global_step, env_config_hash, obs_layout_hash, git_sha}`

**Hata yakalama:**
- Her güncellemede `torch.isfinite(loss)` kontrol edilir. NaN çıkarsa: güncelleme atlanır, son checkpoint yüklenir, lr × 0.5 yapılır ve uyarı verilir.
- `log_std` `clamp` ile sınırlanır.
- `obs_rms` yalnızca veri toplama sırasında güncellenir, güncelleme adımında dondurulur.
- Şekil kontrolleri `assert` ile yapılır.
- `torch.manual_seed`, `np.random.default_rng` ve `torch.use_deterministic_algorithms(True)` testlerde açıktır.

**Testler ve DoD** (`pytest racing_rl/rl`, yalnız CPU, Unity'siz):

1. Şekil ve dtype testleri (tablo). `evaluate` çıkışı [B].
2. `log_prob` ve entropi, `torch.distributions.Normal` referansıyla ≤ 1e-6 farkla eşleşir.
3. GAE: el ile hesaplanmış T = 5 örnek (1 terminated, 1 truncated) ve yavaş referans döngüsüyle ≤ 1e-6 farkla eşleşir.
4. **Backward:** Sentetik batch (B = 1024, rastgele obs, u, adv, returns) ile:
   - loss sonlu
   - aktör, kritik ve `log_std` gradyanlarının tümü sonlu ve ≠ 0
   - `step()` sonrasında parametreler değişir
   - clip sonrası gradyan normu ≤ 0.5
5. **PPO değişmezi:** İlk epoch'un ilk minibatch'inde ρ ≡ 1 (±1e-6), `clip_frac = 0`, `approx_kl ≈ 0`. Bu test, toplama ile değerlendirme arasındaki normalizasyon uyumsuzluğunu yakalar.
6. `RunningMeanStd` paralel güncellemesi, numpy ile birleştirilmiş veri üzerinden hesaplanan ortalama ve varyansla ≤ 1e-6 bağıl farkla eşleşir.
7. **Öğrenme sağlaması:** `TargetReachVecEnv` (26 boyut, hedef ilk 2 boyutta, r = −‖a − hedef‖²). 30 güncelleme içinde ortalama ödül > −0.05 olmalı (CPU'da < 60 s).
8. Checkpoint kaydet-yükle döngüsü: aynı girdiye aynı çıktı. Aynı tohumla ilk güncelleme metrikleri birebir aynı.
9. *(Opsiyonel, `@slow`)* `Pendulum-v1` 1M adımda ortalama dönüş ≥ −250.
