# PixelHDM 架構設計文檔

> **PixelHDM**: Pixel Home-scale Diffusion Model (像素家用規模擴散模型)

**版本**: 1.1.0
**更新日期**: 2026-01-08

---

## 1. 系統概覽

PixelHDM 是一個基於雙路徑 Transformer 的圖像生成模型，結合:
- **PixelDiT**: 雙路徑架構 (Patch級 + Pixel級)
- **DINOv3 REPA**: 特徵對齊損失
- **Triple Loss**: V-Loss + Frequency Loss + REPA Loss

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
│ Patch Embedding (Bottleneck):            │
│ (B, H, W, 3) → unfold → (B, L, 768)     │
│ → Linear(768→256) → SiLU → Linear(256→1024) │
│ → (B, L, D), L = (H/p)×(W/p), D = 1024  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Time Embedding: t → (B, 1024)            │
│ Sinusoidal(256) → Linear(256→1024)       │
│ → SiLU → Linear(1024→1024)               │
│ 時間採樣: Logit-Normal                    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Joint Text-Image Sequence               │
│ (B, T+L, D) = concat(text_embed, img)    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Patch Transformer Blocks × N (16層)      │
│ ├─ TokenAdaLN (時間條件)                  │
│ ├─ pre/post_attn_norm (RMSNorm)         │
│ ├─ GatedMultiHeadAttention (GQA 4:1)    │
│ ├─ pre/post_mlp_norm (RMSNorm)          │
│ └─ SwiGLU FFN                            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Pixel Embedding (1×1 Patchify)           │
│ (B, H, W, 3) → (B, L, p², D_pix)         │
│ 直接從輸入圖像，保留高頻細節              │
│ p² = 256, D_pix = 16                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Pixel Transformer Blocks × M (4層)       │
│ ├─ PixelwiseAdaLN (時間條件)             │
│ ├─ TokenCompaction (Compress-Attend-Expand) │
│ └─ SwiGLU FFN                            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Pixel Patchify + Output Projection       │
│ (B, L, p², D_pix) → (B, H, W, 3)         │
└─────────────────────────────────────────┘
    ↓
輸出預測 (B, H, W, 3)
```

### 2.2 Token Compaction

Compress-Attend-Expand 流程，實現 p⁴ = 65,536× 注意力成本降低:

```
輸入: (B, L, p², D_pix)
       ↓
┌─────────────────────────────────────────┐
│ Compress: Linear(p² × D_pix → D)         │
│ (B, L, 4096) → (B, L, 1024)              │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Pre-Norm: RMSNorm(D)                     │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Attention: GatedMultiHeadAttention (GQA) │
│ 使用 mRoPE 位置編碼                       │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Post-Norm: RMSNorm(D)                    │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Expand: Linear(D → p² × D_pix)           │
│ (B, L, 1024) → (B, L, 4096)              │
│ 使用 xavier_uniform(gain=0.1) 初始化     │
└─────────────────────────────────────────┘
       ↓
輸出: (B, L, p², D_pix) + 殘差連接
```

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

---

## 3. 訓練系統

### 3.1 PixelHDM Flow Matching

時間方向: t=0 噪聲, t=1 乾淨 (與標準相反)

```python
# 時間採樣 (SD3/PixelHDM 參數)
u ~ Normal(μ=0.0, σ=1.0)
t = t_eps + (1 - 2×t_eps) × sigmoid(u)  # t ∈ [0.05, 0.95]

# 插值
z_t = t * x_clean + (1 - t) * noise

# 目標
v_target = x_clean - noise
```

### 3.2 Triple Loss System

```
L = L_vloss + λ_freq × L_freq + λ_repa × L_REPA

L_vloss: 速度空間 MSE
L_freq:  DCT 頻率權重損失 (JPEG Q=90)
L_REPA:  DINOv3 cosine similarity (250K步早停)
```

### 3.3 分桶系統

AspectRatioBucket 動態生成分辨率桶:
- 最小: 256×256
- 最大: 1024×1024 (可配置)
- 步進: patch_size × 4 = 64
- 桶數量: 動態計算 (取決於 min/max/step 配置)

採樣器類別:
- `BucketSampler` (random): 完全隨機
- `SequentialBucketSampler` (sequential): 順序處理 (RAM優化)
- `BufferedShuffleBucketSampler` (buffered_shuffle): 緩衝區優化 (推薦)

---

## 4. 推理系統

### 4.1 採樣流程

```
1. 文本編碼: prompt → Qwen3TextEncoder → (B, T, 1024)
2. 噪聲初始化: z_0 ~ N(0, I) @ t=t_eps
3. 採樣循環 (t: t_eps → 1-t_eps):
   timesteps = linspace(t_eps, 1-t_eps, num_steps+1)  # [0.05, 0.95]
   for t in timesteps:
       v_pred = PixelHDM(z_t, t, text_embed)
       v_cfg = apply_cfg(v_uncond, v_cond, scale)
       z_{t+1} = sampler.step(z_t, v_cfg, t)
4. 後處理: clip(z_1, 0, 1) → PIL.Image
```

### 4.2 採樣器

| 方法 | 階數 | 精度 | 速度 |
|------|------|------|------|
| Euler | 1階 | 一般 | 最快 |
| Heun | 2階 | 較好 | 中等 |
| DPM++ | 高階 | 最好 | 較慢 |
| DPM++ 2S | 高階 | 最好 | 更慢 |

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

### 7.2 推薦配置

```yaml
# 訓練
use_flash_attention: true
use_gradient_checkpointing: true
mixed_precision: bf16

# 推理
sampler_method: heun
num_steps: 50
guidance_scale: 7.5
```

---

## 版本歷史

| 版本 | 日期 | 變更 |
|------|------|------|
| 1.1.0 | 2026-01-08 | 重命名 PixelDiT → PixelHDM |
| 1.0.0 | 2025-12-30 | 初始版本 |

---

**注意**: 此文檔反映 2026-01-08 的代碼狀態
