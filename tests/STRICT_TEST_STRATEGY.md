# 嚴格測試策略 (2026-01-02)

## 目標
以最嚴格的態度創建測試，不信任過去文檔，直接從源代碼分析潛在缺陷。

---

## 已識別的關鍵測試缺口

### 1. Flow Matching 數學正確性 (CRITICAL)

**源文件**: `src/training/flow_matching.py`

**問題發現**:
- `PixelHDMFlowMatching.__init__` 默認參數 `p_mean=-0.8, p_std=0.8` (JiT Class-Conditional)
- 但 T2I 任務應使用 `p_mean=0.0, p_std=1.0` (SD3/PixelHDM 標準)
- 如果 config 為 None，會使用錯誤的默認值

**需要的嚴格測試**:
```python
# test_flow_matching_strict.py

1. test_default_params_without_config
   - 驗證無 config 時 p_mean=-0.8, p_std=0.8 (記錄行為)
   - 這是潛在的設計缺陷，測試應該捕捉

2. test_config_overrides_defaults
   - 驗證 config.time_p_mean=0.0 確實覆蓋默認值
   - 驗證 config.time_p_std=1.0 確實覆蓋默認值

3. test_timestep_distribution_shape
   - P_mean=0.0 時: t>0.8 比例 ~5%
   - P_mean=-0.8 時: t>0.8 比例 ~0.1%
   - 統計驗證分佈形狀

4. test_interpolation_formula
   - z_t = t * x + (1 - t) * noise
   - 當 t=0: z_t = noise
   - 當 t=1: z_t = x

5. test_v_theta_calculation
   - v_theta = (x_pred - z_t) / (1 - t)
   - 驗證數值穩定性 (t 接近 1 時)

6. test_v_target_calculation
   - v_target = x - noise
   - 驗證公式正確

7. test_loss_computation
   - L = MSE(v_theta, v_target)
   - 驗證損失值合理範圍
```

---

### 2. 配置系統 YAML 到代碼傳遞 (CRITICAL)

**源文件**: `src/config/model_config.py`

**問題發現**:
- YAML `flow_matching.P_mean` 需要映射到 `model_data["time_p_mean"]`
- 複雜的嵌套配置解析容易遺漏字段
- 已發生過配置不生效的問題

**需要的嚴格測試**:
```python
# test_config_strict.py

1. test_flow_matching_yaml_parsing
   - 創建 YAML: flow_matching: { P_mean: 0.0, P_std: 1.0 }
   - 驗證 Config.from_yaml 正確解析到 model.time_p_mean

2. test_training_config_betas_parsing
   - YAML: training.optimizer.betas: [0.9, 0.999]
   - 驗證 TrainingConfig.betas 正確解析為 tuple

3. test_zclip_config_parsing
   - YAML: training.gradient.zclip_threshold: 2.5
   - 驗證 TrainingConfig.zclip_threshold 正確解析

4. test_data_config_reference
   - YAML: data_config: "data_config.yaml"
   - 驗證外部數據配置正確加載

5. test_output_section_parsing
   - YAML: output.max_checkpoints: 2
   - 驗證 TrainingConfig.max_checkpoints 正確解析

6. test_config_validation_failure
   - 測試無效配置被拒絕
   - hidden_dim 不能被 num_heads 整除
```

---

### 3. 梯度流與權重初始化 (CRITICAL)

**源文件**: `src/models/pixeldit.py`, `src/models/layers/embedding.py`

**問題發現**:
- 之前發生過 output_proj 零初始化導致梯度斷流
- 修復為 `xavier_uniform_(gain=0.02)`
- 需要嚴格測試確保問題不再發生

**需要的嚴格測試**:
```python
# test_gradient_flow_strict.py

1. test_output_proj_not_zero_initialized
   - 驗證 output_proj.weight 不為全零
   - 驗證權重分佈合理

2. test_full_model_gradient_flow
   - 前向傳播後反向傳播
   - 統計有梯度的參數數量
   - 驗證 > 95% 參數有非零梯度

3. test_gradient_magnitude_reasonable
   - 梯度不應該過大 (>100) 或過小 (<1e-10)
   - 檢測梯度消失/爆炸

4. test_patch_blocks_gradient_flow
   - 驗證每個 patch block 都有梯度

5. test_pixel_blocks_gradient_flow
   - 驗證每個 pixel block 都有梯度

6. test_adaln_gradient_flow
   - AdaLN 條件調製有梯度
```

---

### 4. 損失函數數學公式 (HIGH)

**源文件**: `src/training/losses/`

**需要的嚴格測試**:
```python
# test_losses_strict.py

1. test_vloss_formula
   - V-Loss = MSE(v_theta, v_target)
   - 驗證與手動計算一致

2. test_vloss_zero_when_perfect
   - 當 x_pred 完美時，損失應為 0

3. test_freq_loss_dct_weights
   - DCT 權重計算正確性
   - JPEG Q=90 對應的權重

4. test_freq_loss_ycbcr_conversion
   - RGB 到 YCbCr 轉換正確

5. test_repa_loss_cosine_similarity
   - REPA Loss 使用 cosine similarity
   - 驗證範圍 [0, 2]

6. test_combined_loss_weights
   - 驗證權重正確應用
   - total = vloss + λ_freq * freq + λ_repa * repa
```

---

### 5. Trainer 訓練邏輯 (HIGH)

**源文件**: `src/training/trainer.py`

**需要的嚴格測試**:
```python
# test_trainer_strict.py

1. test_text_encoder_caption_encoding
   - 傳入 captions，驗證 text_embeddings 被正確生成
   - 驗證 text_encoder 被調用

2. test_gradient_accumulation
   - 驗證每 N 步才更新權重
   - 驗證梯度正確累積

3. test_zclip_from_config
   - 驗證 ZClip 參數從配置讀取
   - 不是硬編碼的 2.5

4. test_ema_update
   - 驗證 EMA 權重正確更新
   - 驗證 decay 參數生效

5. test_oom_recovery
   - 模擬 OOM 錯誤
   - 驗證 safe_train_step 正確處理
   - 驗證批次減半邏輯

6. test_checkpoint_cleanup
   - 驗證 max_checkpoints 限制生效
   - 驗證舊檢查點被刪除

7. test_warmup_lr
   - 驗證 warmup 期間 LR 線性增加
   - 驗證 warmup 完成後 initial_lr 正確設置

8. test_lr_scheduler_restart
   - 驗證 cosine_restart 正確重啟
   - 驗證 T_0 計算使用 optimizer 步數
```

---

### 6. 採樣器 ODE 正確性 (HIGH)

**源文件**: `src/inference/sampler.py`, `src/training/flow_matching.py`

**需要的嚴格測試**:
```python
# test_sampler_strict.py

1. test_euler_step
   - z_{t+1} = z_t + dt * v
   - 驗證公式正確

2. test_heun_step
   - 二階預測-校正
   - 驗證比 Euler 更精確

3. test_cfg_formula
   - x_cfg = x_uncond + scale * (x_cond - x_uncond)
   - 驗證 scale=1 時等於 x_cond

4. test_timesteps_direction
   - PixelHDM: t 從 0 (噪聲) 到 1 (乾淨)
   - 驗證 timesteps 遞增

5. test_sampling_from_noise
   - 從純噪聲開始採樣
   - 驗證輸出合理

6. test_deterministic_sampling
   - 相同種子產生相同結果
```

---

### 7. 數據系統 E2E (MEDIUM)

**源文件**: `src/training/dataset/`, `src/training/bucket/`

**需要的嚴格測試**:
```python
# test_data_e2e_strict.py

1. test_yaml_data_dir_used
   - 驗證 YAML data_dir 被 DataLoader 實際使用
   - 不是硬編碼路徑

2. test_bucket_sampler_state_save_restore
   - 保存採樣器狀態
   - 恢復後繼續正確順序

3. test_bucket_resolution_constraint
   - bucket_max_resolution 被遵守
   - 不生成超過限制的分辨率
```

---

### 8. 模型輸入輸出格式 (MEDIUM)

**源文件**: `src/models/pixeldit.py`

**需要的嚴格測試**:
```python
# test_pixeldit_io_strict.py

1. test_input_format_bhwc
   - 輸入 (B, H, W, 3) 正確處理

2. test_input_format_bchw
   - 輸入 (B, 3, H, W) 正確轉換

3. test_output_format
   - 輸出始終為 (B, H, W, 3)

4. test_variable_resolution
   - 256x256, 512x512, 512x256 都正確處理

5. test_repa_layer_index
   - repa_align_layer 正確 (1-indexed 轉 0-indexed)
```

---

## 測試文件結構

```
tests/
├── strict/                          # 嚴格測試目錄
│   ├── test_flow_matching_strict.py # Flow Matching 嚴格測試
│   ├── test_config_strict.py        # 配置系統嚴格測試
│   ├── test_gradient_flow_strict.py # 梯度流嚴格測試
│   ├── test_losses_strict.py        # 損失函數嚴格測試
│   ├── test_trainer_strict.py       # 訓練器嚴格測試
│   ├── test_sampler_strict.py       # 採樣器嚴格測試
│   ├── test_data_e2e_strict.py      # 數據系統 E2E 測試
│   └── test_pixeldit_io_strict.py   # 模型 I/O 測試
```

---

## 執行順序

1. **Flow Matching 測試** - 核心數學正確性
2. **配置系統測試** - 確保配置正確傳遞
3. **梯度流測試** - 確保模型可訓練
4. **損失函數測試** - 確保損失計算正確
5. **訓練器測試** - 確保訓練邏輯正確
6. **採樣器測試** - 確保推理正確
7. **數據系統測試** - 確保數據流正確
8. **模型 I/O 測試** - 確保格式處理正確

---

## 關鍵原則

1. **不信任文檔** - 直接從源代碼驗證
2. **數學嚴格** - 手動計算對照
3. **邊界測試** - 極端值、邊界條件
4. **回歸測試** - 已修復的問題不再發生
5. **E2E 驗證** - 配置到執行的完整鏈路

---

*Created: 2026-01-02*
