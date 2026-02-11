# PixelHDM API 參考手冊

> **PixelHDM**: Pixel Home-scale Diffusion Model (像素家用規模擴散模型)

**版本**: 1.5.2
**更新日期**: 2026-02-11

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
config = PixelHDMConfig.default()    # 默認配置 (~332M參數)
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
| text_processor_layers | int | 2 | TextProcessor 層數 (0=禁用) |
| image_processor_layers | int | 2 | ImageProcessor 層數 (0=禁用) |
| num_heads | int | 16 | Q注意力頭數 |
| num_kv_heads | int | 4 | KV注意力頭數 |
| zero_init_output | bool | True | 是否將 output projection 權重初始化為 0 |
| token_compaction_expand_gain | float | 0.1 | TokenCompaction Expand Xavier 初始化 gain |
| text_processor_mlp_ratio | float \| None | None (→ mlp_ratio) | TextProcessor MLP 比例 |
| image_processor_mlp_ratio | float \| None | None (→ mlp_ratio) | ImageProcessor MLP 比例 |
| text_processor_use_qk_norm | bool | True | TextProcessor 的 QK Norm |
| image_processor_use_qk_norm | bool | True | ImageProcessor 的 QK Norm |
| image_processor_use_timestep | bool | True | ImageProcessor 是否使用 t_embed (TokenAdaLN) |
| pixel_rope_type | "mrope" \| "rope2d" | "rope2d" | Pixel path RoPE 類型 |
| pixel_rope_max_size | int \| None | None | Pixel path RoPE2D max_size (None=沿用 mRoPE max_h/w) |
| use_dynamic_timestep_shift | bool | True | 啟用動態時間步偏移 (DTS) |
| timestep_shift_type | "exponential" \| "linear" | "exponential" | DTS 模式 |
| timestep_shift_fixed | float | 3.0 | exponential 模式固定 shift factor (SD3-style) |
| timestep_shift_base_shift | float | 0.5 | linear 模式最小 shift |
| timestep_shift_max_shift | float | 1.15 | linear 模式最大 shift |
| timestep_shift_base_seq_len | int | 256 | linear 模式最小 token 數 |
| timestep_shift_max_seq_len | int | 4096 | linear 模式最大 token 數 |
| timestep_shift_clamp_linear | bool | True | linear 模式超出區間時 clamp |

**處理器說明**: TextProcessor / ImageProcessor 為 joint self-attention 前的模態內處理。TextProcessor 不注入 RoPE/timestep；ImageProcessor 不注入 RoPE，但預設使用 timestep (TokenAdaLN)，可由 `image_processor_use_timestep=False` 關閉。設為 `0` 可禁用。

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
    # Augmentation control (2026-01-22)
    random_flip=True,           # 隨機水平翻轉
    center_crop=True,           # 中心裁切 (True 會禁用 random_crop)
    use_random_crop=True,       # 隨機裁切 (center_crop=True 時會被禁用)
    default_caption="",         # 預設標題
    caption_dropout=0.1,        # 標題丟棄率
)
```

**重要**: `center_crop=True` 會自動禁用 `use_random_crop`，適用於單圖過擬合測試。

**訓練警示 (實測)**: 若 `zero_init_output=True`，訓練早期可能出現收斂變慢或停滯。建議訓練時設為 `False`，推理或對齊特定初始化需求時才使用。

**DTS 設定**: 在 `configs/train_config.yaml` 的 `flow_matching.dynamic_timestep_shift` 中配置，訓練與推理使用同一組設定。

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
    return_features: bool = False,        # 是否返回中間特徵 (REPA 用)
    return_aux: bool = False,             # 是否返回輔助輸出 (gamma_l2)
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # 返回 (B, H, W, 3) 預測 velocity
    # 若 return_features=True, 返回 (output, features)
    # 若 return_aux=True, 返回 (output, features_or_None, gamma_l2)
```

### 2.2 編碼器

#### create_text_encoder_from_config

```python
from src.models.encoders import create_text_encoder_from_config

text_encoder = create_text_encoder_from_config(config)
hidden_states, attention_mask = text_encoder(
    texts=["a beautiful sunset"],
    return_pooled=False,
)
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
    gamma_l2=gamma_l2,        # PixelwiseAdaLN gamma L2 懲罰 (可選)
)
# loss_dict = {'total': ..., 'vloss': ..., 'freq_loss': ..., 'repa_loss': ..., 'gamma_l2': ...}
# 注意: repa_loss / gamma_l2 始終返回，禁用時值為 0.0
# freq_loss 在 image 空間計算 (x_pred = v_pred + noise, x_clean)，非 velocity 空間
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

#### src.training.train 的 Thread Prefetch（單執行緒）

`python -m src.training.train` 訓練入口在 `num_workers=0` 時，預設啟用 `Thread Prefetch`，
目標是在單執行緒資料路徑下減少資料準備抖動造成的 GPU utilization 波動。

**預設行為**:
- 啟用條件: `num_workers == 0` 且未指定 `--disable-thread-prefetch`
- 預設緩衝區大小: `gradient_accumulation_steps * 2`
- 若 `num_workers > 0`，自動跳過 `Thread Prefetch`（不包裝 `DataLoader`）

**CLI 參數**:
- `--enable-thread-prefetch`: 顯式啟用（預設即為啟用）
- `--disable-thread-prefetch`: 顯式停用
- `--prefetch-buffer-multiplier`: 緩衝倍率（預設 `2`）
- `--prefetch-buffer-items`: 指定絕對緩衝數（優先於 multiplier）

```bash
# 預設行為（num_workers=0 時啟用，buffer=grad_accum*2）
python -m src.training.train --config configs/train_config.yaml

# 關閉 Thread Prefetch
python -m src.training.train --config configs/train_config.yaml --disable-thread-prefetch

# 調整倍率
python -m src.training.train --config configs/train_config.yaml --prefetch-buffer-multiplier 4

# 直接指定絕對緩衝數（優先）
python -m src.training.train --config configs/train_config.yaml --prefetch-buffer-items 8
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
    save_interval=2500,               # 每 N 個 optimizer steps 保存
    save_every_epochs=1,              # 每個 epoch 邊界也保存
    callback=my_callback,             # 可選
    use_progress_bar=True,
)
```

**Checkpoint 與步數語義 (v1.5.1)**:
- `num_epochs` 模式下，每個 epoch 固定為 `len(dataloader)` 個 optimizer steps。
- 梯度積累只影響每 step 的計算量與耗時，不會把每輪 step 數減半。
- 週期保存觸發條件: `step % save_interval == 0` 或 `epoch % save_every_epochs == 0`。
- 若同一 step 同時命中兩種條件，只保存一次（避免重複檔案）。
- 週期檔名統一為 `checkpoint_epoch{epoch}_step{step}.pt`；訓練完成另存 `checkpoint_completed.pt`。

#### safe_train_step (OOM 恢復)

帶 CUDA OOM 自動恢復的訓練步驟。

```python
from src.training import Trainer

trainer = Trainer(model, training_config=training_config)

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
1. 每次 OOM: 清理顯存並嘗試重試
2. 若可行，將批次大小減半後重試
3. 超過最大重試次數或批次無法再減: 返回 None

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
    output_type="pil",              # pil/tensor/numpy
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
| negative_prompt | str | None | 負面提示詞 |
| height | int | 512 | 圖像高度 |
| width | int | 512 | 圖像寬度 |
| num_steps | int | 50 | 採樣步數 |
| guidance_scale | float | 7.5 | CFG引導強度 |
| guidance_rescale | float | 0.0 | Rescaled CFG 強度 (0=關閉) |
| use_dynamic_cfg | bool | False | 是否啟用動態 CFG |
| cfg_schedule | str | "constant" | 動態 CFG 排程 (constant/linear/cosine/quadratic) |
| cfg_min_scale | float | 1.0 | 動態 CFG 最小值 |
| cfg_max_scale | float | None | 動態 CFG 最大值 (None=使用 guidance_scale) |
| num_images_per_prompt | int | 1 | 每提示詞生成圖數 |
| seed | int | None | 隨機種子 |
| sampler_method | str | "heun" | 採樣方法 (euler/heun/dpm_pp/dpm_pp_2s) |
| output_type | str | "pil" | 輸出格式 (pil/tensor/numpy) |
| return_intermediates | bool | False | 是否返回中間步驟 |
| callback | Callable | None | 進度回調函數 |

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
    t_eps=0.0001,
    use_dynamic_timestep_shift=True,  # 默認啟用 DTS
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
| 1.5.2 | 2026-02-11 | 新增訓練入口的 single-thread `Thread Prefetch`（預設 2x）；支援 disable/override `CLI`；`num_workers>0` 時自動跳過以避免與 worker prefetch 疊加 |
| 1.5.1 | 2026-02-09 | 統一訓練步數與梯度積累語義（epoch 步數不因梯度積累減半）；checkpoint 週期保存規則整合（step/epoch 同步命中時去重）；檔名統一為 `checkpoint_epoch{epoch}_step{step}.pt`，最終完成檔保留 `checkpoint_completed.pt` |
| 1.5.0 | 2026-02-07 | **審計修復**: DPM++ DTS 相容性修復; t_eps 統一為 0.0001; timestep_shift_fixed 1.0→3.0; freq_loss 改為 image 空間; Heun 末步 Euler 優化; SamplerConfig DTS 預設啟用; repa_loss 安全存取 |
| 1.4.3 | 2026-02-05 | 新增動態時間步偏移 (DTS) 配置與訓練/推理一致性 |
| 1.4.2 | 2026-02-04 | 新增圖像/文字處理器 |
| 1.4.1 | 2026-01-22 | **回滾修復**: 恢復 DINOv3 ImageNet 正規化; DataLoader augmentation 配置傳遞; mRoPE per-sample text_len + 顯式 max_h/w 配置 |
| 1.4.0 | 2026-01-21 | **關鍵修復**: expand_gain 調整; DINOv3 ImageNet 正規化; DataLoader 配置修復; mRoPE 改進 |
| 1.3.0 | 2026-01-19 | **關鍵修復**: 移除 PixelwiseAdaLN 的 cond_norm; text_max_length 改為 511 |
| 1.2.0 | 2026-01-08 | 重命名 PixelDiT → PixelHDM |
| 1.1.0 | 2025-12-31 | 新增工廠方法 (`create_*_from_config`)、OOM 恢復 (`safe_train_step`) |
| 1.0.0 | 2025-12-30 | 初始版本 |

---

**注意**: 完整參數列表請參考 `src/config/pixelhdm_config.py`
