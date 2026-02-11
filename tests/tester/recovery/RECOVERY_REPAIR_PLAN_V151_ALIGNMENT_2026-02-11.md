# V1.5.1 對齊修復計畫 (2026-02-11)

## 目標
在不做性能掛點、不中斷既有訓練語義的前提下，修復並對齊以下四個檔案與 `publicdocs` 契約：
- `src/training/trainer/step.py`
- `src/training/trainer/loop.py`
- `src/training/trainer/core.py`
- `src/training/train.py`

## 範圍與限制
- 僅修復功能正確性與版本契約一致性（v1.5.1 / v1.5.0）。
- 不新增性能量測欄位、不改 API 表面、不做大重構。
- 禁用破壞性 git 操作（不得使用 `reset --hard`、不得覆蓋未提交變更）。
- 每階段只改一個模組，改完立即驗證。

## 依據文件（權威來源）
1. `publicdocs/IMPLEMENTATION.md`
2. `publicdocs/api/API_REFERENCE.md`
3. `publicdocs/architecture/ARCHITECTURE.md`

## 已確認問題（基線）

### D1 梯度積累語義未對齊（高風險）
- 現況：loop 每步只取單 batch；step 仍做 `loss / gradient_accumulation_steps` 且每步都 `optimizer.step()`。
- 風險：`gradient_accumulation_steps > 1` 時等效梯度縮小，且與文件「accumulation 後才 step」不一致。
- 依據：
  - `publicdocs/IMPLEMENTATION.md:500`
  - `publicdocs/IMPLEMENTATION.md:505`
  - `publicdocs/api/API_REFERENCE.md:298`
- 測試證據：`tests/training/trainer/test_gradient_accumulation.py::TestGradientAccumulation::test_drop_last_false_updates_on_tail_micro_batches`

### D2 scheduler 同步步數公式不一致（高風險）
- 現況：`core.py` 仍以 `state.step // gradient_accumulation_steps` 同步 scheduler。
- 風險：若 `state.step` 已為 optimizer-step，resume 後 LR 週期錯位。
- 依據：
  - `publicdocs/IMPLEMENTATION.md:503`
  - `publicdocs/IMPLEMENTATION.md:505`
  - `publicdocs/architecture/ARCHITECTURE.md:422`

### D3 `repa_loss` 安全存取缺失（高風險）
- 現況：`step.py` metrics 仍使用 `loss_dict["repa_loss"]` 直接索引。
- 風險：在 simple-loss 路徑可觸發 `KeyError`。
- 依據：
  - `publicdocs/api/API_REFERENCE.md:223`
  - `publicdocs/IMPLEMENTATION.md:923`

### D4 `len(dataloader)` 不可用時策略不明確（中風險）
- 現況：epoch mode 走到 `len(self.dataloader)` 可能直接 `TypeError`。
- 風險：未符合文件要求的 fail-fast 規則與可診斷錯誤訊息。
- 依據：
  - `publicdocs/IMPLEMENTATION.md:508`
  - `publicdocs/IMPLEMENTATION.md:509`
  - `publicdocs/IMPLEMENTATION.md:510`

### D5 resume 時 step/epoch mismatch 缺 warning（中風險）
- 現況：loop 會重算 epoch，但缺明確 mismatch 警告。
- 依據：
  - `publicdocs/IMPLEMENTATION.md:507`

## 實施策略（分階段）

## Phase 0：凍結基線（只讀）
目標：先固定失敗面，不修改 `src`。

執行：
- `py -3.10 -m pytest tests/training/trainer/test_gradient_accumulation.py -q -x`
- `py -3.10 -m pytest tests/training/trainer/test_step_with_simple_loss.py -q`
- `py -3.10 -m pytest tests/training/trainer/test_loop.py -q -x`
- `py -3.10 -m pytest tests/training/trainer/test_scheduler_factory.py -q -x`

退出條件：
- 輸出可重現，失敗點映射到 D1-D5。

## Phase 1：修復 `step.py`（先處理 D3，再處理 accumulation 介面）
目標：確保 loss 欄位安全存取，並提供與 loop 協同的 accumulation 入口。

修改重點：
- `repa_loss` 讀取改為 `.get("repa_loss", 0.0 tensor)`。
- 保持 `gamma_l2` 一致的安全存取模式。
- 若採用 micro-batch accumulation：
  - 新增/恢復 `forward_backward(..., denom)` 與 `optimizer_step_and_metrics(...)` 的分離式流程。
  - 僅在 optimizer boundary 觸發 optimizer/scheduler/ema。

驗證：
- `py -3.10 -m pytest tests/training/trainer/test_step.py -q`
- `py -3.10 -m pytest tests/training/trainer/test_step_with_simple_loss.py -q`

退出條件：
- `step` 相關測試通過，且無 `KeyError: repa_loss`。

## Phase 2：修復 `loop.py`（D1/D4/D5 主體）
目標：落實 v1.5.1 step/epoch 語義與 fail-fast 規則。

修改重點：
- 引入 micro-batch 收集與 `drop_last_accumulation` 策略。
- epoch boundary 改為 step-driven（`step % len(dataloader) == 0`）。
- `len(dataloader)` 不可用時：
  - epoch mode 直接 `ValueError`（明確訊息）。
  - steps mode 僅允許 epoch hooks 關閉。
- resume mismatch 時記錄 warning（step/epoch 重對齊證據）。

驗證：
- `py -3.10 -m pytest tests/training/trainer/test_gradient_accumulation.py -q -x`
- `py -3.10 -m pytest tests/training/trainer/test_loop.py -q`

退出條件：
- gradient accumulation 與 loop 契約測試綠燈。

## Phase 3：修復 `core.py`（D2）
目標：scheduler 同步使用正確 step 單位。

修改重點：
- 若 `state.step` 定義為 optimizer-step，移除 `// gradient_accumulation_steps` 的再縮放。
- 保持 warmup/scheduler 皆以同一單位運作。

驗證：
- `py -3.10 -m pytest tests/training/trainer/test_scheduler_factory.py -q`
- `py -3.10 -m pytest tests/training/trainer/test_gradient_accumulation.py::TestGradientAccumulation::test_scheduler_step_matches_optimizer_step_count -q`

退出條件：
- scheduler 步數與 optimizer-step 一致。

## Phase 4：校正 `train.py`（四檔收斂）
目標：確認入口層不破壞上層語義。

修改重點：
- 僅做必要對齊：
  - 不引入性能掛點參數。
  - 透傳與 config 對齊的訓練控制參數（保持現有對外行為）。
- 日誌訊息需對應實際保存策略（step + epoch）。

驗證：
- `py -3.10 -m pytest tests/training/test_trainer.py -q`
- `py -3.10 -m pytest tests/strict/test_trainer_strict.py -q -x`

退出條件：
- 入口層與 trainer/loop 語義一致，無新回歸。

## Phase 5：回歸與收斂
目標：確認四檔修復後無側向破壞。

驗證包：
- `py -3.10 -m pytest tests/training/trainer/test_step.py tests/training/trainer/test_loop.py tests/training/trainer/test_gradient_accumulation.py -q`
- `py -3.10 -m pytest tests/training/test_trainer.py -q`
- `py -3.10 tests/tester/recovery/run_recovery_smoke.py`

退出條件：
- 目標測試全綠。
- 無新增 strict 類失敗（至少抽樣 `trainer_strict`）。

## 風險控制
- 單檔單階段提交（本地變更層面，不要求 git commit）。
- 每階段先測再改再測，不跨階段堆疊未驗證改動。
- 出現非預期新失敗時立即停下，先做 root-cause 再前進。

## 驗收定義
- D1-D5 全部關閉。
- 與 `publicdocs` v1.5.1/v1.5.0 條目無衝突。
- 四檔修復不引入性能掛點、不更動既有公開接口。

## 執行紀錄
使用 append-only 日誌：
- `tests/tester/recovery/RECOVERY_EXEC_LOG_2026-02-11.md`

每條紀錄格式：
```text
[timestamp UTC]
phase: Phase N
command: <exact command>
result: PASS | FAIL
failing_tests:
- <test id>
top_stack:
- <path:line> <message>
notes:
- <D1-D5 mapping>
```
