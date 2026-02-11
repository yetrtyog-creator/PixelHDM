# 四檔最嚴格審查計畫 (v1.5.1 對齊)

日期: 2026-02-11  
範圍: 僅審查，不做性能掛點，不擴張需求。

## 1. 審查目標

針對以下四檔做「契約級」審查，確認與 `publicdocs` 的 v1.5.1 / v1.5.0 行為完全一致:
- `src/training/trainer/step.py`
- `src/training/trainer/core.py`
- `src/training/trainer/loop.py`
- `src/training/train.py`

## 2. 權威依據 (唯一判準)

- `publicdocs/IMPLEMENTATION.md`
- `publicdocs/architecture/ARCHITECTURE.md`
- `publicdocs/api/API_REFERENCE.md`

關鍵版本條目:
- v1.5.1 (2026-02-09): 梯度積累下 epoch-step 語義統一、step/epoch checkpoint 觸發整合且同一步去重、檔名統一 `checkpoint_epoch{epoch}_step{step}.pt`
- v1.5.0 (2026-02-07): `repa_loss` 安全存取、DTS 相容性修復、`t_eps=0.0001`、Heun 末步 Euler 等

## 3. 契約矩陣 (必驗)

`C01` epoch-step 語義  
- 依據: `publicdocs/IMPLEMENTATION.md` (Step Semantics v1.5.1)  
- 必須成立: epoch 模式每輪固定 `len(dataloader)` 個 optimizer steps，不能被 `gradient_accumulation_steps` 減半。

`C02` 積累只影響每 step 計算量  
- 依據: `publicdocs/api/API_REFERENCE.md` 3.4  
- 必須成立: accumulation 只改有效 batch 與每 step 時長，不改 epoch 的 step 計數。

`C03` epoch 邊界以 step 驅動  
- 依據: `publicdocs/IMPLEMENTATION.md`  
- 必須成立: 邊界判定為 `step % len(dataloader) == 0`，不是物理 `StopIteration`。

`C04` resume 對齊規則  
- 依據: `publicdocs/IMPLEMENTATION.md`  
- 必須成立: 以 `epoch = step // len(dataloader)` 對齊，且 mismatch 時有警告證據。

`C05` len 不可用時 fail-fast/降級規則  
- 依據: `publicdocs/IMPLEMENTATION.md`  
- 必須成立:
- epoch mode: 直接錯誤並可診斷。
- steps mode: 僅在 epoch-based save/log 全關閉時允許。

`C06` checkpoint 觸發去重  
- 依據: `publicdocs/api/API_REFERENCE.md`、`publicdocs/architecture/ARCHITECTURE.md`  
- 必須成立: step 觸發與 epoch 觸發同一步命中時只保存一次。

`C07` checkpoint 命名契約  
- 依據: 三份文件一致條目  
- 必須成立:
- 週期檔: `checkpoint_epoch{epoch}_step{step}.pt`
- 完成檔: `checkpoint_completed.pt`

`C08` `repa_loss` / `gamma_l2` 安全存取  
- 依據: `publicdocs/api/API_REFERENCE.md` 3.1、v1.5.0  
- 必須成立: loss dict 缺鍵時不得 KeyError，需退回 0.0。

`C09` 文字編碼契約 (廢棄 pooled 依賴)  
- 依據: `publicdocs/api/API_REFERENCE.md` 2.2  
- 必須成立: 編碼路徑使用 `return_pooled=False` 並以 hidden_states/mask 契約工作，不依賴 `pooled_text_embed`。

`C10` 入口層參數透傳  
- 依據: `publicdocs/api/API_REFERENCE.md` 3.4  
- 必須成立: `train.py -> Trainer.train(...)` 正確透傳 `save_every_epochs`、`log_every_epochs`，且日誌描述與實際策略一致。

## 4. 四檔審查清單

## 4.1 `src/training/trainer/step.py`
- 檢查 `forward_backward` 與 `optimizer_step_and_metrics` 是否遵守「先累積、後一步 optimizer」。
- 檢查 `_backward_scaled(..., denom)` 與分母來源是否一致。
- 檢查 `_create_metrics_from_accum` 是否對 `repa_loss/gamma_l2` 使用安全 fallback。
- 檢查 `_encode_captions` 回傳 tuple/dict 的兼容分支與 `return_pooled=False`。

## 4.2 `src/training/trainer/core.py`
- 檢查 scheduler sync 單位是否為 optimizer-step (`state.step`)。
- 檢查 `train_step` 對 list micro-batches 的處理是否只做一次 optimizer step。
- 檢查 `safe_train_step` 在 OOM/retry 路徑不破壞 accumulator 狀態。

## 4.3 `src/training/trainer/loop.py`
- 檢查 `_collect_micro_batches` 與 `drop_last_accumulation` 邏輯。
- 檢查 `_sync_epoch_with_step` 邊界判定與 epoch 遞增。
- 檢查 `_validate_unknown_len_policy` 是否嚴格實作 C05。
- 檢查 `_handle_periodic_checkpoint` 的去重與命名契約。

## 4.4 `src/training/train.py`
- 檢查 `parse_args` 與現行需求是否一致 (不可混入性能掛點參數)。
- 檢查 `trainer.train(...)` 透傳與儲存策略日誌。
- 檢查例外路徑 (KeyboardInterrupt/Exception) 是否保留 checkpoint 行為。

## 5. 證據採集與量化

## 5.1 靜態證據
```powershell
rg -n "forward_backward|optimizer_step_and_metrics|repa_loss|gamma_l2|return_pooled=False" src/training/trainer/step.py
rg -n "_sync_scheduler_to_step|optimizer_steps|state.step|gradient_accumulation_steps" src/training/trainer/core.py
rg -n "_collect_micro_batches|_sync_epoch_with_step|_validate_unknown_len_policy|checkpoint_epoch|checkpoint_completed" src/training/trainer/loop.py
rg -n "save_every_epochs|log_every_epochs|trainer.train\\(" src/training/train.py
```

## 5.2 動態證據 (必要測試集)
```powershell
py -3.10 -m pytest tests/training/trainer/test_step.py -q
py -3.10 -m pytest tests/training/trainer/test_step_with_simple_loss.py -q
py -3.10 -m pytest tests/training/trainer/test_loop.py -q
py -3.10 -m pytest tests/training/trainer/test_gradient_accumulation.py -q
py -3.10 -m pytest tests/training/test_trainer.py -q
py -3.10 -m pytest tests/training/trainer/test_scheduler_factory.py -q
```

## 5.3 量化判準
- 契約通過率: `10/10` (C01~C10 全通過才可結案)。
- 測試: 上述必要測試全綠。
- 命名稽核: 週期 checkpoint 名稱零違規。
- 例外稽核: 無 `KeyError('repa_loss')`、無 scheduler step 單位錯位證據。

## 6. 失敗分級與修復優先序

- `P0` (立即阻斷): C01/C03/C06/C07/C08 任一失敗。
- `P1` (高優先): C04/C05/C10 失敗。
- `P2` (次優先): C09 失敗或日誌/描述不一致。

修復順序固定:
1. `step.py` (`C08` + 積累接口)
2. `loop.py` (`C01/C03/C04/C05/C06/C07`)
3. `core.py` (`C02` + scheduler 單位)
4. `train.py` (`C10`)

## 7. 執行紀錄格式 (append-only)

將每次審查或修復記錄到:
- `tests/tester/recovery/RECOVERY_EXEC_LOG_2026-02-11.md`

記錄模板:
```text
[UTC timestamp]
phase: strict_audit
contract: C0X
target: <file>
command: <exact command>
result: PASS | FAIL
evidence:
- <file:line or test id>
risk: P0|P1|P2
action: keep | patch-required
```

## 8. 凍結規則

- 審查期間禁止新增性能量測掛點。
- 禁止擅自擴張到非四檔修改。
- 若發現與文件衝突，先記錄證據再修復，不可先改後補證據。
