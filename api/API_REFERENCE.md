# PixelHDM API 參考手冊

> **PixelHDM**: Pixel Home-scale Diffusion Model (像素家用規模擴散模型)

**版本**: 1.3.0
**更新日期**: 2026-01-19

---

## 目錄

1. [配置 API](#1-配置-api)
2. [模型 API](#2-模型-api)
3. [訓練 API](#3-訓練-api)
4. [推理 API](#4-推理-api)

---

## 1. 配置 API

### src.config.PixelHDMConfig

主要模型配置類。

```python
from src.config import PixelHDMConfig

# 使用工廠方法創建
config = PixelHDMConfig.default()    # 默認配置 (~357M參數)
config = PixelHDMConfig.small()      # 測試用小配置
config = PixelHDMConfig.large()      # 大型配置
config = PixelHDMConfig.for_testing() # 單元測試配置
```

**核心參數**:

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| hidden_dim | int | 1024 | 主隱藏維度 |
| pixel_dim | int | 16 | 像素特徵維度 |
| patch_size | int | 16 | Patch大小 |
| patch_layers | int | 16 | Patch級層數 |
| pixel_layers | int | 4 | 像素級層數 |
| num_heads | int | 16 | Q注意力頭數 |
| num_kv_heads | int | 4 | KV注意力頭數 |

### src.config.TrainingConfig

訓練超參數配置。

```python
from src.config import TrainingConfig

training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    max_steps=500000,
)
```

### src.config.DataConfig

數據集與分桶配置。

```python
from src.config import DataConfig

data_config = DataConfig(
    use_bucketing=True,
    min_bucket_size=256,
    max_bucket_size=1024,
    sampler_mode="buffered_shuffle",
)
```

---

## 2. 模型 API

### 2.1 主模型

#### create_pixelhdm_for_t2i

創建完整的文本到圖像模型。

```python
from src.models import create_pixelhdm_for_t2i

model = create_pixelhdm_for_t2i(
    config=None,              # 可選，使用 PixelHDMConfig.default()
    load_text_encoder=True,   # 是否加載文本編碼器
    load_dino_encoder=True,   # 是否加載 DINO 編碼器
)
model.cuda()
```

#### create_pixelhdm_from_config

從配置創建基礎 PixelHDM 模型。

```python
from src.models.pixelhdm import create_pixelhdm_from_config
from src.config import PixelHDMConfig

config = PixelHDMConfig.default()
model = create_pixelhdm_from_config(config)
```

#### create_pixelhdm_for_t2i_from_config

從配置創建文本到圖像模型。

```python
from src.models.pixelhdm import create_pixelhdm_for_t2i_from_config
from src.config import PixelHDMConfig

config = PixelHDMConfig.default()
model = create_pixelhdm_for_t2i_from_config(
    config,
    load_text_encoder=True,
    load_dino_encoder=True,
)
```

#### PixelHDM.forward

模型前向傳播。

```python
def forward(
    x_t: torch.Tensor,                    # (B, H, W, 3) 帶噪圖像
    t: torch.Tensor,                      # (B,) 時間步 [0, 1]
    text_embed: torch.Tensor = None,      # (B, T, D) 文本嵌入
    text_mask: torch.Tensor = None,       # (B, T) 文本掩碼
    pooled_text_embed: torch.Tensor = None,  # (B, D) 池化文本嵌入
    return_features: bool = False,        # 是否返回中間特徵 (REPA 用)
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # 返回 (B, H, W, 3) 預測 velocity
    # 若 return_features=True, 返回 (output, features)
```

### 2.2 編碼器

#### create_text_encoder_from_config

```python
from src.models.encoders import create_text_encoder_from_config

text_encoder = create_text_encoder_from_config(config)
embeddings = text_encoder(["a beautiful sunset"])
```

#### create_dinov3_encoder_from_config

```python
from src.models.encoders import create_dinov3_encoder_from_config

dino_encoder = create_dinov3_encoder_from_config(config)
features = dino_encoder(images)  # (B, L, 768)
```

---

## 3. 訓練 API

### 3.1 損失函數

#### create_combined_loss_from_config

```python
from src.training import create_combined_loss_from_config

loss_fn = create_combined_loss_from_config(config)

# 使用
loss_dict = loss_fn(
    v_pred=v_pred,            # 模型預測的 velocity
    x_clean=x_clean,          # 乾淨圖像
    noise=noise,              # 噪聲
    h_t=repa_features,        # 模型中間特徵 (REPA 用，可選)
    step=current_step,        # 當前訓練步數 (REPA early stop 用)
    dino_features=dino_feats, # DINOv3 特徵 (REPA 用，可選)
)
# loss_dict = {'total': ..., 'vloss': ..., 'freq_loss': ..., 'repa_loss': ...}
# 注意: repa_loss 始終返回，禁用時值為 0.0
```

### 3.2 Flow Matching

#### create_flow_matching

```python
from src.training import create_flow_matching

flow_matching = create_flow_matching(config)

# 準備訓練數據
t, z_t, x, noise = flow_matching.prepare_training(x_clean)
```

#### create_flow_matching_from_config

從配置創建 PixelHDM Flow Matching 模組。

```python
from src.training.flow_matching import create_flow_matching_from_config
from src.config import PixelHDMConfig

config = PixelHDMConfig.default()
flow_matching = create_flow_matching_from_config(config)
```

### 3.3 數據加載

#### create_dataloader_from_config_v2

推薦的數據加載方式，支持分桶。

```python
from src.training import create_dataloader_from_config_v2

dataloader = create_dataloader_from_config_v2(
    root_dir="./data/train",
    model_config=config.model,
    data_config=config.data,
)
```

### 3.4 訓練器

#### Trainer

```python
from src.training import Trainer, create_trainer

# 創建訓練器 (dataloader 在初始化時傳入)
trainer = create_trainer(
    model=model,
    config=model_config,              # PixelHDMConfig
    training_config=training_config,  # TrainingConfig
    dataloader=dataloader,            # 可選
    device=device,                    # 可選
    text_encoder=text_encoder,        # 可選，用於 CFG
)

# 訓練
trainer.train(
    num_steps=500000,                 # 或 num_epochs=16
    log_interval=100,
    save_interval=1000,
    save_every_epochs=0,              # 默認值為 0 (不按 epoch 保存)
    callback=my_callback,             # 可選
    use_progress_bar=True,
)
```

#### safe_train_step (OOM 恢復)

帶 CUDA OOM 自動恢復的訓練步驟。

```python
from src.training import Trainer

trainer = Trainer(model, training_config=config)

# 普通訓練步驟
metrics = trainer.train_step(batch)

# 帶 OOM 恢復的訓練步驟
metrics = trainer.safe_train_step(
    batch,
    retry_on_oom=True,   # 是否在 OOM 時重試
    max_retries=3,       # 最大重試次數
)

if metrics is None:
    print("訓練步驟失敗（OOM 經過重試仍無法恢復）")
```

**OOM 恢復邏輯**:
1. 第一次 OOM: 清理顯存並重試
2. 第二次 OOM: 減半批次大小後重試
3. 超過最大重試次數: 返回 None

---

## 4. 推理 API

### 4.1 推理管線

#### PixelHDMPipeline

完整的文本到圖像推理管線。

```python
from src.inference import PixelHDMPipeline

pipeline = PixelHDMPipeline(model)

output = pipeline(
    prompt="a beautiful sunset over the ocean",
    negative_prompt=None,           # 負面提示詞
    height=512,
    width=512,
    num_steps=50,
    guidance_scale=7.5,
    num_images_per_prompt=1,        # 每提示詞生成圖數
    seed=42,
    sampler_method="heun",          # euler/heun/dpm_pp/dpm_pp_2s
    output_type="pil",              # pil/tensor
    return_intermediates=False,     # 返回中間步驟
    callback=None,                  # 進度回調函數
)

output.images[0].save("output.png")
```

#### create_pipeline_from_config

從配置創建完整推理管線。

```python
from src.inference.pipeline import create_pipeline_from_config
from src.config import PixelHDMConfig
import torch

config = PixelHDMConfig.default()
pipeline = create_pipeline_from_config(
    config,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
    load_text_encoder=True,
    load_dino_encoder=False,  # 推理時不需要 DINO
)

output = pipeline("a beautiful sunset")
```

**參數說明**:

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| prompt | str | 必需 | 文本提示 |
| height | int | 512 | 圖像高度 |
| width | int | 512 | 圖像寬度 |
| num_steps | int | 50 | 採樣步數 |
| guidance_scale | float | 7.5 | CFG引導強度 |
| seed | int | None | 隨機種子 |
| sampler_method | str | "heun" | 採樣方法 (euler/heun/dpm_pp/dpm_pp_2s) |

### 4.2 採樣器

#### create_sampler_from_config

```python
from src.inference import create_sampler_from_config

sampler = create_sampler_from_config(config)

# 或直接創建
from src.inference import UnifiedSampler

sampler = UnifiedSampler(
    method="heun",
    num_steps=50,
    t_eps=0.05,
)
```

### 4.3 CFG

#### StandardCFG

```python
from src.inference import StandardCFG, apply_cfg

# 方式 1: 使用 apply_cfg 函數
output = apply_cfg(
    x_cond=cond_output,
    x_uncond=uncond_output,
    guidance_scale=7.5,
    rescale_factor=0.0,  # 可選，Rescaled CFG
)

# 方式 2: 使用 StandardCFG 類
cfg = StandardCFG()
output = cfg.apply(
    x_cond=cond_output,
    x_uncond=uncond_output,
    guidance_scale=7.5,
)
```

---

## 快速開始

### 訓練

```python
from src.config import Config
from src.models import create_pixelhdm_for_t2i
from src.training import create_dataloader_from_config_v2, create_trainer

# 1. 配置
config = Config.from_yaml("configs/train_config.yaml")

# 2. 模型
model = create_pixelhdm_for_t2i(config=config.model)
model.cuda()

# 3. 數據
train_loader = create_dataloader_from_config_v2(
    root_dir="./data",
    model_config=config.model,
    data_config=config.data,
)

# 4. 訓練器 (loss_fn 內部創建)
trainer = create_trainer(
    model=model,
    config=config.model,
    training_config=config.training,
    dataloader=train_loader,
)

# 5. 訓練
trainer.train(num_steps=500000)  # 或 num_epochs=16
```

### 推理

```python
from src.models import create_pixelhdm_for_t2i
from src.inference import PixelHDMPipeline

# 1. 加載模型
model = create_pixelhdm_for_t2i()
model.cuda()

# 2. 創建管線
pipeline = PixelHDMPipeline(model)

# 3. 生成
output = pipeline(
    prompt="a beautiful sunset",
    height=512,
    width=512,
)
output.images[0].save("output.png")
```

---

## 版本歷史

| 版本 | 日期 | 變更 |
|------|------|------|
| 1.3.0 | 2026-01-19 | **CRITICAL**: 移除 PixelwiseAdaLN 的 cond_norm; text_max_length 改為 511 |
| 1.2.0 | 2026-01-08 | 重命名 PixelDiT → PixelHDM |
| 1.1.0 | 2025-12-31 | 新增工廠方法 (`create_*_from_config`)、OOM 恢復 (`safe_train_step`) |
| 1.0.0 | 2025-12-30 | 初始版本 |

---

**注意**: 完整參數列表請參考 `src/config/pixelhdm_config.py`
