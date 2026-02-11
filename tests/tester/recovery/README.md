# Recovery Workspace

## Documents
- Requirements matrix:
  - `tests/tester/recovery/RECOVERY_REQUIREMENTS_2026-02-11.md`
- Repair execution plan:
  - `tests/tester/recovery/RECOVERY_REPAIR_PLAN_2026-02-11.md`
  - `tests/tester/recovery/RECOVERY_REPAIR_PLAN_V151_ALIGNMENT_2026-02-11.md`
  - `tests/tester/recovery/STRICT_AUDIT_PLAN_4FILES_V151_2026-02-11.md`
- Execution log (append-only):
  - `tests/tester/recovery/RECOVERY_EXEC_LOG_2026-02-11.md`

## Smoke Checks

Run:

```powershell
py -3.10 tests\tester\recovery\run_recovery_smoke.py
```

Purpose:
- Validate critical recovery contracts with a fast, non-invasive script.
- Complements unit tests by focusing on high-risk cross-cutting behavior.

Current checks:
- `prepare_batch_contract`
- `cfg_dropout_errors`
- `metrics_zero_safe`
- `loop_epoch_step_boundary`
- `loop_checkpoint_contracts`
