# PixelHDM 架構設計文檔

> **PixelHDM**: Pixel Home-scale Diffusion Model (像素家用規模擴散模型)

**版本**: 1.5.2
**更新日期**: 2026-02-11

---

## 1. 系統概覽

PixelHDM 是一個基於雙路徑 Transformer 的圖像生成模型，結合:
- **PixelHDM**: 雙路徑架構 (Patch級 + Pixel級)
- **DINOv3 REPA**: 特徵對齊損失
- **Triple Loss**: V-Loss + Frequency Loss + REPA Loss (+ optional γ-L2)

### 1.1 目錄結構

```
src/
├── config/                 # 配置管理 [350行]
│   ├── pixelhdm_config.py  # 模型配置
│   ├── pixelhdm_factories.py # 配置工廠
│   └── ...
├── models/                 # 模型核心 [2,500行]
│   ├── pixelhdm/           # 主模型目錄
│   │   ├── core.py         # PixelHDM 核心
│   │   └── t2i.py          # PixelHDMForT2I
│   ├── attention/          # 注意力機制
│   ├── blocks/             # Transformer塊
│   ├── encoders/           # 編碼器
│   └── layers/             # 基礎層
├── training/               # 訓練系統 [3,200行]
│   ├── flow_matching/      # PixelHDM Flow Matching
│   ├── losses/             # 損失函數
│   └── optimization/       # 優化工具
└── inference/              # 推理系統 [1,650行]
    ├── pipeline/           # 推理管線
    ├── sampler/            # 採樣器
    └── cfg/                # CFG 策略
```

### 1.2 依賴關係

```
config/ ← 無依賴 (配置中心)
   ↑
models/ ← 依賴 config
   ↑
training/ ← 依賴 config, models
   ↑
inference/ ← 依賴 config, models, training
```

---

## 2. 核心模型架構

### 2.1 PixelHDM 雙路徑設計

```
輸入圖像 (B, H, W, 3)
    ↓

┌─────────────────────────────────────────┐
│ Patch Embedding (Bottleneck)            │
│ (B,H,W,3) -> unfold -> (B,L,768)        │
│ -> Linear(768->bottleneck) -> SiLU      │
│ -> Linear(bottleneck->1024)             │
│ -> (B,L,D), L=(H/p)x(W/p), D=1024       │
│ bottleneck_dim = patch_size^2 // 4      │
│ (p=16 -> 64)                            │
└─────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────┐
│ ImageProcessor x2 (pre-joint)           │
│ (B,L,D) -> (B,L,D)                      │
│ image self-attn, no RoPE; optional t         │
└─────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────┐
│ Text Encoder + TextProjector            │
│ prompt -> text_embed -> (B,T,D)         │
└─────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────┐
│ TextProcessor x2 (pre-joint)            │
│ (B,T,D) -> (B,T,D)                      │
│ text self-attn, no RoPE/t               │
└─────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────┐
│ Time Embedding: t -> (B,1024)           │
│ Sinusoidal(256) -> Linear(256->1024)    │
│ -> SiLU -> Linear(1024->1024)           │
│ t from Flow Matching (Logit-Normal)     │
└─────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────┐
│ Joint Text-Image Sequence               │
│ (B,T+L,D) = concat(text_proc, img_proc) │
└─────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────┐
│ Patch Transformer Blocks × N (16layer)  │
│ ├─ TokenAdaLN (Time conditions)         │
│ ├─ pre/post_attn_norm (RMSNorm)         │
│ ├─ GatedMultiHeadAttention (GQA 4:1)    │
│ ├─ pre/post_mlp_norm (RMSNorm)          │
│ └─ SwiGLU FFN                           │
└─────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────┐
│ Pixel Embedding (1×1 Patchify)          │
│ (B, H, W, 3) → (B, L, p², D_pix)        │
│ High-frequency details are obtained     │
│  directly from the input image.         │
│ p² = 256, D_pix = 16                    │
└─────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────┐
│ Pixel Transformer Blocks × M (4layer)   │
│ ├─ PixelwiseAdaLN (Time+TextConditions) │
│ │  ⚠️ no cond_norm                      │
│ ├─ TokenCompaction(CompressAttendExpand, RoPE2D)│
│ └─ SwiGLU FFN                           │
└─────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────┐
│ Output Norm + Pixel Patchify            │
│ RMSNorm(D_pix) → (B, L, p², D_pix)      │
│ → (B, H, W, 3)                          │
│ 可選 zero_init_output:output_proj零初始化 │
└─────────────────────────────────────────┘

    ↓

輸出預測 (B, H, W, 3)
```

**補充**:
- TextProcessor / ImageProcessor 為模態內雙層處理，僅在 joint self-attention 前使用。
- RoPE 僅在 joint blocks 注入；pixel path image-only 注意力使用 RoPE2D；TextProcessor 不注入 timestep；ImageProcessor 使用 TokenAdaLN 注入 timestep（可關閉）。

### 2.2 Token Compaction

Compress-Attend-Expand 流程，實現 p⁴ = 65,536× 注意力成本降低:

```
輸入: (B, L, p², D_pix)
       ↓
┌─────────────────────────────────────────┐
│ Compress: Linear(p² × D_pix → D)        │
│ (B, L, 4096) → (B, L, 1024)             │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Pre-Norm: RMSNorm(D)                    │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Attention: GatedMultiHeadAttention (GQA)│
│ 使用 mRoPE 位置編碼                       │
│ ⚠️ 無內部殘差 (符合架構圖設計)              │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Post-Norm: RMSNorm(D)                   │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│ Expand: Linear(D → p² × D_pix)                              │
│ (B, L, 1024) → (B, L, 4096)                                 │
│ 使用 xavier_uniform(gain=token_compaction_expand_gain)初始化  │
└─────────────────────────────────────────────────────────────┘
       ↓
輸出: (B, L, p², D_pix)
       ↓
殘差連接由外層 PiT Block 通過 α₁ gating 處理:
x_out = x + α₁ × TokenCompaction(modulate(x))
```

**關鍵設計 (2026-01-20 修正)**:
- **MHSA 內部無殘差**: 架構圖中 Compress → MHSA → Expand 是直線流過
- **殘差在 Block 層級**: 通過 α₁ gate 控制，符合 adaLN-Zero 設計
- **初始化一致性**: Expand 權重在全局初始化後會再依 `token_compaction_expand_gain` 重設

### 2.3 mRoPE 多軸旋轉位置編碼

維度分配 (head_dim=64):
- 文本: 16維
- 圖像高度: 24維
- 圖像寬度: 24維

```
文本序列: RoPE1D(text_positions, dim=16)
圖像序列: RoPE2D(img_h_positions, img_w_positions, dim=24+24)

合併: concat(text_rope, img_h_rope, img_w_rope)
```

**關鍵配置**:
- `text_max_length = 511` (非 512)
- `mrope_text_max_len = 511` (非 512)
- 原因: 圖像 token 的 axis0 使用 `text_len` 作為位置，若 text_len=512 會越界

**2026-01-22 改進**:
- `mrope_img_max_height = 128`: 顯式配置最大高度 (取代 sqrt 計算)
- `mrope_img_max_width = 128`: 顯式配置最大寬度 (支援極端長寬比)
- `create_position_ids_batched` 現支援 per-sample text_len:
  - 接收 `text_mask` 參數
  - 為每個樣本計算實際文本長度
  - 圖像 token 使用 per-sample 的 actual_text_len 作為 axis0

### 2.4 PixelwiseAdaLN 條件化機制

PixelwiseAdaLN 將時間和文字條件融入 Pixel Transformer:

```
s_cond (B, L, 1024) ── 包含語義、時間、文字信息
    │
    ▼
cond_expand: Linear(1024 → 4096), Xavier uniform
    │
    ▼
reshape: (B, L, 4096) → (B, L, 256, 16)
    │
    ▼
param_gen: SiLU + Linear(16 → 96) ── 生成 6 組 AdaLN 參數
    │
    ▼
輸出: gamma1, beta1, alpha1, gamma2, beta2, alpha2 (各 16 維)
```

**⚠️ 關鍵設計 (2026-01-19)**:
- **無 cond_norm**: 曾添加 RMSNorm 嘗試恢復信號，但它破壞了 99.7% 的文字條件
- **問題**: cond_expand 將不同文字的 s_cond 投影到幾乎平行的向量 (cosine_sim=0.998)
- **RMSNorm 只保留方向** → 歸一化後不同文字輸入變得相同
- **信號保留**: 有 cond_norm=0.3%, 無 cond_norm=54%

---

## 3. 訓練系統

### 3.1 PixelHDM Flow Matching

時間方向: t=0 噪聲, t=1 乾淨 (與標準相反)

```python
# 時間採樣 (SD3/PixelHDM 參數)
u ~ Normal(μ=0.0, σ=1.0)
u' = p_mean + p_std × u  # 默認 p_mean=0.0, p_std=1.0
t = t_eps + (1 - 2×t_eps) × sigmoid(u')  # t ∈ [t_eps, 1 - t_eps]

# 插值
z_t = t * x_clean + (1 - t) * noise

# 目標
v_target = x_clean - noise
```

### 3.2 Dynamic Timestep Shift (DTS)

訓練與推理一致的時間步偏移，依解析度調整采樣分佈：

- 以 `sigma = 1 - t` 作為偏移基準
- DTS 只作用在 `sigma`，再回寫 `t' = 1 - sigma'`
- 依 `num_tokens` 計算 shift（線性或指數）

**訓練/推理一致性要求**:
- 相同解析度必須使用相同 `shift_config` 與 `t_eps`
- 若解析度不能整除 `patch_size`，建議跳過 DTS 並記錄警告
- `timestep_shift_fixed` 預設 3.0 (SD3-style)；設為 1.0 會使 exponential 模式無效 (log(1)=0)
- 所有 sampler 均支援 DTS 參數 (含 DPM++)

### 3.3 Triple Loss System

```
L = L_vloss + λ_freq × L_freq + λ_repa × L_REPA + λ_gamma × L_gamma (可選)

L_vloss: 速度空間 MSE
L_freq:  DCT 頻率權重損失 (JPEG Q=90) — 在 image 空間計算 (x_pred, x_clean)
L_REPA:  DINOv3 cosine similarity (250K步早停)
L_gamma: PixelwiseAdaLN gamma L2 penalty (pixel_gamma_l2_lambda > 0 時啟用)
```

### 3.4 分桶系統

AspectRatioBucket 動態生成分辨率桶:
- 最小: 256×256
- 最大: 1024×1024 (可配置)
- 步進: patch_size × 4 = 64
- 桶數量: 動態計算 (取決於 min/max/step 配置)

採樣器類別:
- `BucketSampler` (random): 完全隨機
- `SequentialBucketSampler` (sequential): 順序處理 (RAM優化)
- `BufferedShuffleBucketSampler` (buffered_shuffle): 緩衝區優化 (推薦)

### 3.5 單執行緒 Thread Prefetch（Training Entry）

為了在 Windows `num_workers=0` 路徑下減少資料等待抖動，訓練入口加入 `Thread Prefetch`
（Producer/Consumer + bounded queue）：

```
Base DataLoader (main thread iterator)
    ↓
ThreadPrefetchDataLoader
    ├─ Producer thread: 持續生產下一批資料
    ├─ Bounded queue: 緩衝區（預設 grad_accum × 2）
    └─ Consumer (training loop): 從 queue 消費 batch
```

**啟用策略**:
- 預設啟用條件: `num_workers == 0` 且未指定停用 flag
- CLI 可顯式停用: `--disable-thread-prefetch`
- `num_workers > 0` 時自動跳過（避免與多 worker prefetch 疊加）

**穩定性設計**:
- Producer 例外會透傳到 Consumer（不吞錯）
- 以 sentinel 結束訊號終止 iterator，避免卡死
- 訓練結束時呼叫 `close()` 回收背景 thread

---

## 4. 推理系統

### 4.1 採樣流程

```
1. 文本編碼: prompt → Qwen3TextEncoder → (B, T, 1024)
2. 噪聲初始化: z_0 ~ N(0, I) @ t=t_eps
3. 採樣循環 (t: t_eps → 1-t_eps):
   timesteps = get_timesteps(t_eps, num_steps, use_logit_normal=True)
   # 若啟用 DTS，先轉換 sigma 再回寫 t
   # 若 use_logit_normal=False，則使用 linspace 均勻時間步
   z = z_0
   for i in range(num_steps):
       t = timesteps[i]
       t_next = timesteps[i + 1]
       z = sampler.step(
           model=PixelHDM, z=z, t=t, t_next=t_next,
           text_embeddings=cond, null_text_embeddings=uncond,
           guidance_scale=scale, guidance_rescale=guidance_rescale
       )
4. 後處理: clip(z, 0, 1) → PIL.Image
```

### 4.2 採樣器

| 方法 | 階數 | 精度 | 速度 |
|------|------|------|------|
| Euler (`euler`) | 1階 | 一般 | 最快 |
| Heun (`heun`) | 2階 (末步 Euler) | 較好 | 中等 |
| DPM++ (`dpm_pp`) | 高階 | 最好 | 較慢 |
| DPM++ 2S (`dpm_pp_2s`) | 高階 | 最好 | 更慢 |

**Heun 末步優化**: Heun sampler 在最後一步自動退化為 Euler (跳過 corrector)，節省 1 NFE。50 步 Heun 實際 NFE = 2×50 - 1 = 99。

### 4.3 CFG 策略

```python
# Standard CFG
x_cfg = x_uncond + scale * (x_cond - x_uncond)

# Rescaled CFG (推薦)
x_cfg = x_uncond + scale * (x_cond - x_uncond)
factor = std(x_cond) / (std(x_cfg) + 1e-8)  # +1e-8 防止除零
x_out = rescale_factor * (x_cfg * factor) + (1 - rescale_factor) * x_cfg
```

---

## 5. 設計決策

### 5.1 為什麼使用 DINOv3

- patch_size=16 與 PixelHDM 完美匹配 (無需插值)
- 比 DINOv2 (patch_size=14) 更適合本架構
- **嚴禁回退到 DINOv2**

### 5.2 為什麼使用 PixelHDM 時間方向

- t=0 噪聲, t=1 乾淨 (與標準相反)
- 更穩定的訓練
- 更好的低噪聲區域建模

### 5.3 為什麼使用 GQA 4:1

- 減少 KV 緩存 75%
- 與 Qwen3 架構一致
- 保持注意力質量

---

## 6. 擴展指南

### 6.1 添加新損失函數

1. 在 `src/training/losses/` 創建 `new_loss.py`
2. 實現 `NewLoss(nn.Module)` 類
3. 在 `CombinedLoss` 中添加
4. 在 `PixelHDMConfig` 中添加配置參數

### 6.2 添加新採樣器

1. 在 `src/inference/sampler/` 創建 `new_sampler.py` 類
2. 實現 `step(z_t, t, score_fn)` 方法
3. 在 `create_sampler()` 中添加

### 6.3 修改模型架構

需要更新的位置:
- `PixelHDMConfig`: 新參數
- `src/models/pixelhdm/`: 前向傳播邏輯
- 相關的 Embedding/Block 類

---

## 7. 性能優化

### 7.1 已實現的優化

- Flash Attention: 3-8× 加速
- 梯度檢查點: 50% 顯存減少
- 混合精度 (bf16): 2× 加速
- ZClip: 自適應梯度剪裁
- 梯度積累語義統一: 每個 epoch 固定 `len(dataloader)` 個 optimizer steps，accumulation 只增加單步計算成本

### 7.2 推薦配置

```yaml
# 訓練
use_flash_attention: true
use_gradient_checkpointing: true
mixed_precision: bf16
output:
  save_interval: 2500
  save_every_epochs: 1

# 推理
sampler_method: heun
num_steps: 50
guidance_scale: 7.5
```

Checkpoint 命名規則: `checkpoint_epoch{epoch}_step{step}.pt`。若 step 週期與 epoch 邊界同時命中，僅保存一次。

---

## 版本歷史

| 版本 | 日期 | 變更 |
|------|------|------|
| 1.5.2 | 2026-02-11 | 新增 Training Entry 的 single-thread `Thread Prefetch` 架構說明（預設 2x、`num_workers>0` 自動跳過、Producer 例外透傳與關閉回收流程） |
| 1.5.1 | 2026-02-09 | 統一梯度積累步數語義（每輪 optimizer steps 不因梯度積累減少）；checkpoint 週期保存規則整合並在 step/epoch 同步命中時去重；checkpoint 檔名改為 `checkpoint_epoch{epoch}_step{step}.pt` |
| 1.5.0 | 2026-02-07 | **審計修復**: DPM++ DTS 相容性; t_eps 0.05→0.0001; fixed_shift 1.0→3.0; freq_loss→image 空間; Heun 末步 Euler; SamplerConfig DTS 預設啟用 |
| 1.4.3 | 2026-02-05 | 新增動態時間步偏移 (DTS) 架構與一致性規範 |
| 1.4.2 | 2026-02-04 | 新增圖像/文字處理器 |
| 1.4.1 | 2026-01-22 | **回滾修復**: 恢復 DINOv3 ImageNet 正規化; DataLoader augmentation 配置; mRoPE per-sample text_len + 顯式 max_h/w |
| 1.4.0 | 2026-01-21 | **關鍵修復**: expand_gain 0.1→0.5; DINOv3 ImageNet 正規化; DataLoader 配置修復; mRoPE per-sample text_len |
| 1.3.0 | 2026-01-20 | **關鍵修復**: 移除 TokenCompaction 內部 MHSA 殘差 (符合架構圖設計) |
| 1.2.0 | 2026-01-19 | **關鍵修復**: 移除 PixelwiseAdaLN 的 cond_norm (它破壞 99.7% 文字信號); MRoPE text_max_length 修復 |
| 1.1.0 | 2026-01-08 | 重命名 PixelDiT → PixelHDM |
| 1.0.0 | 2025-12-30 | 初始版本 |

---
