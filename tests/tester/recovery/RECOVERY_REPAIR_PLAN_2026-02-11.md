# Src Recovery Repair Plan (2026-02-11)

## Scope
Restore `src` to expected pre-hookpoint functional behavior (not performance instrumentation), with minimal-risk, test-first execution.

## Constraints
- No destructive git operations.
- No broad refactor.
- Prioritize documented contracts and existing tests over inferred behavior.
- Keep edits limited to:
  - `src/training/trainer/step.py`
  - `src/training/trainer/loop.py`

## Source Priority (Highest -> Lowest)
1. Unit tests:
   - `tests/training/trainer/test_step.py`
   - `tests/training/trainer/test_loop.py`
   - `tests/training/test_trainer.py`
2. Public docs:
   - `publicdocs/api/API_REFERENCE.md`
   - `publicdocs/architecture/ARCHITECTURE.md`
   - `publicdocs/IMPLEMENTATION.md`
3. Recovery hints:
   - `recovery/restored_from_pyc/*.txt`

## Execution Strategy

### Phase 0: Baseline Capture (Read-only)
Goal: lock current failure baseline before edits.

Actions:
- Run:
  - `py -3.10 -m pytest tests/training/trainer/test_step.py -q`
  - `py -3.10 -m pytest tests/training/trainer/test_loop.py -q`
  - `py -3.10 -m pytest tests/training/test_trainer.py -q`
  - `py -3.10 tests/tester/recovery/run_recovery_smoke.py`
- Record failing test IDs and error signatures into recovery notes.

Exit criteria:
- Failure set is reproducible and mapped to requirement IDs (R1-R7).

### Phase 1: StepExecutor Contract Repair (`step.py`)
Goal: repair R1/R2/R3 without changing training semantics.

Change set:
- R1: `_prepare_batch` returns exactly 3 items:
  - `(images, text_embeddings, text_mask)`
- R2: cfg dropout behavior:
  - `cfg_dropout==0` => pass-through.
  - `cfg_dropout>0` + missing `text_encoder` => `ValueError`.
  - `cfg_dropout>0` + missing embeddings => `ValueError`.
  - dropout branch uses null embedding replacement + boolean mask.
- R3: metrics are zero-safe:
  - avoid division by zero in `samples_per_sec`.

Validation after Phase 1:
- `py -3.10 -m pytest tests/training/trainer/test_step.py -q`
- `py -3.10 tests/tester/recovery/run_recovery_smoke.py`

Exit criteria:
- `test_step.py` passes.
- Smoke checks for batch/cfg/zero-safe pass.

### Phase 2: Loop Contract Repair (`loop.py`)
Goal: repair R4/R5/R6/R7 with minimal control-flow changes.

Change set:
- R4: epoch boundary must sync to step modulo dataloader length (not only `StopIteration`).
- R5: checkpoint path fallback:
  - if `save_path is None`, use config `checkpoint_dir`.
- R6: checkpoint name format:
  - `checkpoint_epoch{epoch}_step{step}`.
- R7: dedupe save when step and epoch triggers coincide on same step.

Validation after Phase 2:
- `py -3.10 -m pytest tests/training/trainer/test_loop.py -q`
- `py -3.10 tests/tester/recovery/run_recovery_smoke.py`

Exit criteria:
- `test_loop.py` passes.
- Smoke checks for epoch/checkpoint contracts pass.

### Phase 3: Full Recovery Regression
Goal: ensure no cross-regression after both repairs.

Validation:
- `py -3.10 -m pytest tests/training/trainer/test_step.py tests/training/trainer/test_loop.py tests/training/test_trainer.py -q`
- Optional broader sanity:
  - `py -3.10 -m pytest tests/training -q`

Exit criteria:
- Target suites green.
- No new failures introduced in `test_trainer.py`.

## Precision Measurement and Evidence Collection
For every phase, record:
- command,
- timestamp,
- pass/fail,
- failing test IDs,
- top stack location (`file:line`).

Store in:
- `tests/tester/recovery/RECOVERY_EXEC_LOG_2026-02-11.md` (append-only).

## Risk Controls
- Edit only one module per phase.
- Run tests immediately after each phase; no batching across modules.
- If new unexpected failures appear, stop and triage before further edits.
- Do not introduce new instrumentation fields or API surface during recovery.

## Decision Rules
- If tests and docs conflict: follow tests first, then document mismatch.
- If pyc hint conflicts with tests/docs: treat pyc as non-authoritative.
- If behavior is ambiguous: choose minimal change that satisfies current contracts.

## Done Definition
- R1-R7 satisfied.
- Recovery smoke script fully passes.
- Target unit suites pass consistently on rerun.
- Recovery log contains exact evidence and final status.

## Immediate Next Step
Start Phase 0 baseline capture and create:
- `tests/tester/recovery/RECOVERY_EXEC_LOG_2026-02-11.md`
