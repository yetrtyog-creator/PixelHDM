# PixelHDM

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Experimental](https://img.shields.io/badge/status-experimental-orange.svg)]()

**Pixel Home-scale Diffusion Model**

*A Dual-Path Diffusion Transformer for High-Resolution Image Generation*

[English](#english) | [中文](#中文)

</div>

---

> **Disclaimer / 免責聲明**
>
> This is an **experimental vibe coding project** that has **NOT been fully tested**. It is provided solely for **experimental reference and modification purposes**. Use at your own risk.
>
> 這是一個**實驗性氛圍編碼項目**，**尚未經過充分測試**。本項目僅供**實驗參照修改使用**，風險自負。

---

### References / 參考論文

This project draws inspiration from the following research papers:

| Paper | Link | Description |
|-------|------|-------------|
| **PixelDiT** | [arXiv:2511.20645](https://arxiv.org/abs/2511.20645) | PixelDiT dual-path architecture concept |
| **DiT** | [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) | Scalable Diffusion Models with Transformers |
| **REPA / iREPA** | [arXiv:2512.10794](https://arxiv.org/abs/2512.10794) | Representation Alignment for Generation |
| **Flow Matching** | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) | Flow Matching for Generative Modeling |
| **DINOv3** | [GitHub](https://github.com/facebookresearch/dinov2) | Self-supervised Vision Transformer (Meta 2025) |
| **Gated Attention** |  [arXiv:2505.06708](https://arxiv.org/abs/2505.06708). | Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free |

---

<a name="english"></a>
## English

### Overview

PixelHDM is a state-of-the-art text-to-image generation model combining:

- **PixelDiT**: Dual-path design (Patch-level + Pixel-level) for high-fidelity generation
- **DINOv3 iREPA Loss**: Improved REPA with Conv2d projection and spatial normalization ([arXiv:2512.10794](https://arxiv.org/abs/2512.10794))
- **Triple Loss System**: V-Loss + Frequency Loss + iREPA Loss for comprehensive optimization
- **Flow Matching**: Modern ODE-based generative framework with V-Prediction
- **Gated Attention**: Based on Qwen3-next design.

### Key Features

| Feature | Description |
|---------|-------------|
| **Dual-Path Architecture** | 16×16 Patch path for semantics + 1×1 Pixel path for details |
| **GQA 4:1** | Grouped Query Attention (16 Q heads, 4 KV heads) for efficiency |
| **Token Compaction** | 65,536× attention cost reduction via Compress-Attend-Expand |
| **mRoPE** | Multi-axis Rotary Position Embedding (Lumina2-style) |
| **Flash Attention** | 3-8× speedup with memory efficiency |
| **Multi-Resolution** | Dynamic bucketing from 256×256 to 1024×1024 |

### Architecture

```
Input Image (B, H, W, 3)
         │
         ├──► Patch Embedding (16×16) ──► Patch Transformer (N=16 layers)
         │                                        │
         │                                        ▼
         │                                   s_cond (semantic conditioning)
         │                                        │
         └──► Pixel Embedding (1×1) ─────────────►│
                                                  ▼
                                    Pixel Transformer (M=4 layers)
                                          │
                                          ▼
                                  Output (B, H, W, 3)
```

### Installation

#### Windows (Recommended)

```bash
# Use the setup script (auto-configures Python 3.12 + PyTorch + Flash Attention)
scripts\setup_env.bat

# Activate environment
scripts\activate.bat
```

#### Manual Installation

```bash
# Clone the repository
git clone https://github.com/yetrtyog-creator/PixelHDM.git
cd PixelHDM

# Create virtual environment (Python 3.12 required)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install PyTorch (CUDA 12.8)
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

#### DINOv3 Weights (License Required)

```
DINOv3 requires Meta AI authorization. Please download from:
https://github.com/facebookresearch/dinov2

Place at: Dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

### Quick Start

#### Training

```python
from src.config import Config
from src.models import create_pixelhdm_for_t2i
from src.training import create_dataloader_from_config_v2, create_trainer

# Load config
config = Config.from_yaml("configs/train_config.yaml")

# Create model
model = create_pixelhdm_for_t2i(config=config.model)
model.cuda()

# Create dataloader
dataloader = create_dataloader_from_config_v2(
    root_dir="./data",
    model_config=config.model,
    data_config=config.data,
)

# Create trainer and train
trainer = create_trainer(
    model=model,
    config=config.model,
    training_config=config.training,
    dataloader=dataloader,
)
trainer.train(num_epochs=16)
```

#### Inference

```python
from src.models import create_pixelhdm_for_t2i
from src.inference import PixelHDMPipeline

# Load model
model = create_pixelhdm_for_t2i()
model.cuda()

# Create pipeline
pipeline = PixelHDMPipeline(model)

# Generate image
output = pipeline(
    prompt="a beautiful sunset over the ocean",
    height=512,
    width=512,
    num_steps=50,
    guidance_scale=7.5,
)
output.images[0].save("output.png")
```

#### CLI Inference

```bash
# With trained checkpoint
python -m src.inference.run --prompt "a cat sitting on a windowsill" --steps 50

# With specific checkpoint and dtype
python -m src.inference.run \
    --checkpoint checkpoints/model.pt \
    --prompt "a beautiful landscape" \
    --dtype float16 \
    --sampler heun

# Test pipeline without trained weights (output will be noise)
python -m src.inference.run --random-init --prompt "test"
```

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 1024 | Main hidden dimension |
| `pixel_dim` | 16 | Pixel feature dimension |
| `patch_size` | 16 | Patch size (matches DINOv3) |
| `patch_layers` | 16 | Number of Patch Transformer layers |
| `pixel_layers` | 4 | Number of Pixel Transformer layers |
| `num_heads` | 16 | Number of Q attention heads |
| `num_kv_heads` | 4 | Number of KV attention heads (GQA) |

### Samplers

| Sampler | Order | Quality | Speed | NFE |
|---------|-------|---------|-------|-----|
| `euler` | 1st | Good | Fastest | N |
| `heun` | 2nd | Better | Medium | 2N |
| `dpm_pp` | High | Best | Slower | ~1.5N |

### Project Structure

```
src/
├── config/              # Configuration management
│   └── pixelhdm_config.py
├── models/              # Model core
│   ├── pixelhdm/        # Main model (core.py, t2i.py)
│   ├── attention/       # GQA, Token Compaction
│   ├── blocks/          # Patch/Pixel Transformer blocks
│   ├── encoders/        # DINOv3, Qwen3 text encoder
│   └── layers/          # RoPE, AdaLN, SwiGLU, etc.
├── training/            # Training system
│   ├── flow_matching/   # Flow Matching implementation
│   ├── losses/          # V-Loss, FreqLoss, iREPA
│   ├── trainer/         # Trainer with OOM recovery
│   └── bucket/          # Multi-resolution bucketing
└── inference/           # Inference system
    ├── pipeline/        # Text2Image, Image2Image
    ├── sampler/         # Euler, Heun, DPM++
    └── cfg/             # CFG strategies
```

### Documentation

- [API Reference](api/API_REFERENCE.md) - Complete API documentation
- [Architecture](architecture/ARCHITECTURE.md) - System architecture and design decisions
- [Implementation Details](IMPLEMENTATION.md) - Parameter choices and rationale

### Citation

```bibtex
@software{PixelHDM2026,
  title = {PixelHDM: Pixel Home-scale Diffusion Model},
  year = {2026},
  url = {https://github.com/yetrtyog-creator/PixelHDM}
}
```

---

<a name="中文"></a>
## 中文

### 概述

PixelHDM 是一個結合以下技術的先進文本到圖像生成模型：

- **PixelDiT 架構**：雙路徑設計（Patch級 + Pixel級）實現高保真生成
- **DINOv3 iREPA 損失**：改進的 REPA，採用 Conv2d 投影和空間歸一化（[arXiv:2512.10794](https://arxiv.org/abs/2512.10794)）
- **三重損失系統**：V-Loss + 頻率損失 + iREPA 損失，全面優化
- **Flow Matching**：基於 ODE 的現代生成框架，使用 V-Prediction
- **Gated Attention** :基於Qwen3-next的設計。

### 主要特性

| 特性 | 說明 |
|------|------|
| **雙路徑架構** | 16×16 Patch 路徑處理語義 + 1×1 Pixel 路徑處理細節 |
| **GQA 4:1** | 分組查詢注意力（16 個 Q 頭，4 個 KV 頭）提升效率 |
| **Token 壓縮** | 通過 Compress-Attend-Expand 降低 65,536 倍注意力成本 |
| **mRoPE** | 多軸旋轉位置編碼（Lumina2 風格） |
| **Flash Attention** | 3-8 倍加速，記憶體效率高 |
| **多解析度** | 動態分桶支持 256×256 到 1024×1024 |

### 架構圖

```
輸入圖像 (B, H, W, 3)
         │
         ├──► Patch 嵌入 (16×16) ──► Patch Transformer (N=16 層)
         │                                    │
         │                                    ▼
         │                               s_cond (語義條件)
         │                                    │
         └──► Pixel 嵌入 (1×1) ──────────────►│
                                              ▼
                                Pixel Transformer (M=4 層)
                                      │
                                      ▼
                              輸出 (B, H, W, 3)
```

### 安裝

#### Windows（推薦）

```bash
# 使用安裝腳本（自動配置 Python 3.12 + PyTorch + Flash Attention）
scripts\setup_env.bat

# 啟動環境
scripts\activate.bat
```

#### 手動安裝

```bash
# 克隆倉庫
git clone https://github.com/yetrtyog-creator/PixelHDM.git
cd PixelHDM

# 創建虛擬環境（需要 Python 3.12）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 安裝 PyTorch（CUDA 12.8）
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 安裝依賴
pip install -r requirements.txt
```

#### DINOv3 權重（需要授權）

```
DINOv3 需要 Meta AI 授權，請自行下載：
https://github.com/facebookresearch/dinov2

放置位置：Dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

### 快速開始

#### 訓練

```python
from src.config import Config
from src.models import create_pixelhdm_for_t2i
from src.training import create_dataloader_from_config_v2, create_trainer

# 載入配置
config = Config.from_yaml("configs/train_config.yaml")

# 創建模型
model = create_pixelhdm_for_t2i(config=config.model)
model.cuda()

# 創建數據加載器
dataloader = create_dataloader_from_config_v2(
    root_dir="./data",
    model_config=config.model,
    data_config=config.data,
)

# 創建訓練器並訓練
trainer = create_trainer(
    model=model,
    config=config.model,
    training_config=config.training,
    dataloader=dataloader,
)
trainer.train(num_epochs=16)
```

#### 推理

```python
from src.models import create_pixelhdm_for_t2i
from src.inference import PixelHDMPipeline

# 載入模型
model = create_pixelhdm_for_t2i()
model.cuda()

# 創建管線
pipeline = PixelHDMPipeline(model)

# 生成圖像
output = pipeline(
    prompt="美麗的海上日落",
    height=512,
    width=512,
    num_steps=50,
    guidance_scale=7.5,
)
output.images[0].save("output.png")
```

#### 命令列推理

```bash
# 使用訓練好的檢查點
python -m src.inference.run --prompt "坐在窗台上的貓" --steps 50

# 指定檢查點和數據類型
python -m src.inference.run \
    --checkpoint checkpoints/model.pt \
    --prompt "美麗的風景" \
    --dtype float16 \
    --sampler heun

# 測試管線（無訓練權重，輸出為噪聲）
python -m src.inference.run --random-init --prompt "測試"
```

### 模型配置

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `hidden_dim` | 1024 | 主隱藏維度 |
| `pixel_dim` | 16 | 像素特徵維度 |
| `patch_size` | 16 | Patch 大小（與 DINOv3 匹配） |
| `patch_layers` | 16 | Patch Transformer 層數 |
| `pixel_layers` | 4 | Pixel Transformer 層數 |
| `num_heads` | 16 | Q 注意力頭數 |
| `num_kv_heads` | 4 | KV 注意力頭數（GQA） |

### 採樣器

| 採樣器 | 階數 | 品質 | 速度 | NFE |
|--------|------|------|------|-----|
| `euler` | 一階 | 良好 | 最快 | N |
| `heun` | 二階 | 較好 | 中等 | 2N |
| `dpm_pp` | 高階 | 最佳 | 較慢 | ~1.5N |

### 項目結構

```
src/
├── config/              # 配置管理
│   └── pixelhdm_config.py
├── models/              # 模型核心
│   ├── pixelhdm/        # 主模型 (core.py, t2i.py)
│   ├── attention/       # GQA, Token Compaction
│   ├── blocks/          # Patch/Pixel Transformer 塊
│   ├── encoders/        # DINOv3, Qwen3 文本編碼器
│   └── layers/          # RoPE, AdaLN, SwiGLU 等
├── training/            # 訓練系統
│   ├── flow_matching/   # Flow Matching 實現
│   ├── losses/          # V-Loss, FreqLoss, iREPA
│   ├── trainer/         # 支持 OOM 恢復的訓練器
│   └── bucket/          # 多解析度分桶
└── inference/           # 推理系統
    ├── pipeline/        # Text2Image, Image2Image
    ├── sampler/         # Euler, Heun, DPM++
    └── cfg/             # CFG 策略
```

### 文檔

- [API 參考手冊](api/API_REFERENCE.md) - 完整 API 文檔
- [架構設計](architecture/ARCHITECTURE.md) - 系統架構與設計決策
- [實現細節](IMPLEMENTATION.md) - 參數選擇與設計理由

### 引用

```bibtex
@software{PixelHDM2026,
  title = {PixelHDM: 像素家用規模擴散模型},
  year = {2026},
  url = {https://github.com/yetrtyog-creator/PixelHDM}
}
```

---

<div align="center">

**Made with PyTorch**

*Last Updated: 2026-01-22*

</div>

2026/1/20

修復了歸一化造成的信號衰減與修復增益數值設定錯誤的問題。

Fixed signal attenuation caused by normalization and incorrect gain value settings.

在src/config/pixelhdm_config.py中才可以正確設定adaln_init_gain數值。

The adaln_init_gain value can only be correctly configured in src/config/pixelhdm_config.py.

修正modulate的計算公式錯誤。

Correct the error in the modulate function formula.

移除異常冗餘的殘差設計，這可能會導致訓練模型困難。

Remove abnormal and redundant residual designs, as they may make model training difficult.

2026/1/22

修正REPA行為在桶中會假設方形可能造成特徵扭曲的問題。

Fix the issue where REPA assumes a square bucket shape, which can lead to feature distortion.

同時REPA的image正則化也已實施確保不會出問題。

Meanwhile, REPA image regularization has been implemented to ensure no issues arise.

(已修復) 正在處理修復深度縮放的問題，Agent並沒有發現缺失導致排查困難，最終在完善足夠全面的測試中捕抓到加高層數收斂極其困難的問題，當前正在進行修復與測試。

(Fixed) We are currently addressing an issue with depth scaling. Initial troubleshooting was difficult as the Agent failed to detect the missing components. Ultimately, by implementing comprehensive testing, we identified that convergence becomes extremely difficult as the number of layers increases. Refinement and testing are currently underway.
Remove abnormal and redundant residual designs, as they may make model training difficult.

新增了一直忘記放了config。

Added the missing config.

當前收斂性已得到修正而恢復正常。

Fixed and restored convergence.

drop_last現在並非硬編碼，可以在設置中修改，避免桶中圖片張數不滿足batch size被丟棄而未被訓練。

drop_last is no longer hard-coded and can be modified in the settings. This prevents images from being discarded and excluded from training when the number of images in a bucket doesn't meet the batch size.

如果你需要訓練多種長寬比且收斂好，盡可能均勻各種長寬比數量，並擁有大量詳細有結構的文字描述。

Training for multi-aspect ratio support with optimal convergence requires maintaining an even distribution of various image shapes and leveraging extensive, well-structured captions.

2026/1/23

引入係數k=2，改變深度縮放效果從1/平方根(層數)改變為1/平方根(k*層數)，使得訓練更穩定更好，對於小量樣本收斂效果可能會變得差些。

By introducing a coefficient $k=2$, we modified the depth scaling factor from $1/\sqrt{L}$ to $1/\sqrt{kL}$. This enhances training stability and overall performance, although convergence on smaller datasets may be slightly compromised.

2026/1/24

當前觀測到patch嵌入層瓶頸在64時(嵌入層維度768)為最優，但仍需要足夠的驗證檢查(已經多次檢驗)，大致評估後約為(patch size/2)^2為最優或(patch size^2)/4，整體趨勢與JIT(just image transformer)論文幾乎相同(無論圖片相似度或損失值)。待測試完善後約在1/26部署更新到github上。

觀察到的表現趨勢——特別是 $64 > 128 > 256 > 32 > 16$ 的關係——與 JIT 論文中報告的 FID-50K 曲線高度一致，儘管我是基於當前的相似性指標和損失值而不是實際的 FID-50K 運行進行評估的。

但此結論並不適合用於過低patch size的情況下，例如未滿patch size 8也不太需要使用瓶頸與分層設計，會更偏向一般的DiT設計的改進。

Current observations indicate that a bottleneck size of 64 in the patch embedding layer (with an embedding dimension of 768) yields optimal performance. Although verified multiple times, further rigorous validation is required. Preliminary evaluations suggest the optimal size approximates $(patch\_size/2)^2$, or equivalently $(patch\_size^2)/4$. The overall trend aligns closely with the JIT (Just Image Transformer) paper in terms of both image similarity and loss values. Following the completion of final testing, the update is scheduled for deployment to GitHub on January 26th.

The observed trend in performance—specifically the relationship $64 > 128 > 256 > 32 > 16$—is highly consistent with the FID-50K curves reported in the JIT paper, even though we are evaluating based on current similarity metrics and loss values rather than actual FID-50K runs. 

This conclusion does not hold for extremely small patch sizes. For patch sizes under 8, bottleneck and hierarchical structures are less necessary, as the design tends to follow an improved version of the conventional DiT.

(探討)

傳統DiT基於流匹配在訓練量大時patch size=1或2並沒有太大差別甚至可以說沒差，但在patch size=4以上時會開始特別。
而要探究低維流形有效基本上屬於patch size=8以上的範疇。
同時一些人的研究和測試指出patch size大於8時X預測才優於V預測，但仍要差於更低patch size，難以用訓練量和網路容量彌補。
不同的patch size下可能效果不同，以pixelDiT的研究結論在模型更大時patch size=16跟8是沒有什麼差異的，但是更往上patch size=32時可能會需要相當大容量的模型才能抹平差異，此時瓶頸的真實影響難以估計。

實際大量數據且patch size=32和大模型尤其隱藏層足夠大時的並未充分研究，故patch size=32時patch嵌入層維度3072瓶頸到256，還是需要更高或更低並未知曉，仍然需要有人實驗探索。

(Discussion)

Traditional Diffusion Transformers (DiT) based on Flow Matching show little to no significant difference between patch size=1 or 2 when the training scale is large. However, peculiar characteristics begin to emerge when the patch size reaches 4 or above.

To investigate the effectiveness of the low-dimensional manifold, one generally needs to look at the regime where patch size ≥ 8. Meanwhile, some studies and empirical tests indicate that X-prediction only outperforms V-prediction when the patch size is greater than 8. Nevertheless, its performance remains inferior to that of smaller patch sizes, a gap that is difficult to bridge even with increased training volume or network capacity.

The effects may vary across different patch sizes. According to research on PixelDiT, there is negligible difference between patch size=16 and 8 as the model scales up. However, moving further to patch size=32 may require a significantly larger model capacity to offset the performance drop; at this point, the true impact of the bottleneck becomes difficult to estimate.

The scenario involving massive datasets combined with patch size=32 and large-scale models (especially those with sufficiently large hidden layers) remains under-researched. Consequently, it is still unknown whether a patch embedding dimension of 3072 bottlenecked down to 256 is optimal for patch size=32, or if a higher/lower dimension is required. This area still necessitates further experimental exploration.
