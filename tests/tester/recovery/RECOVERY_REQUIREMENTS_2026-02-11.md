# Src Recovery Requirements (2026-02-11)

## Goal
Reconstruct `src` behavior to the expected pre-hookpoint functional state using:
- public docs as declared requirements
- unit tests as executable contracts
- pyc recovery artifacts as historical hints

## Sources Of Truth
1. `tests/training/trainer/test_step.py`
2. `tests/training/trainer/test_loop.py`
3. `tests/training/test_trainer.py`
4. `publicdocs/api/API_REFERENCE.md` (v1.5.1 notes)
5. `publicdocs/architecture/ARCHITECTURE.md` (v1.5.1 notes)
6. `publicdocs/IMPLEMENTATION.md` sections 5.6/5.9/5.11
7. `recovery/restored_from_pyc/step_dump.txt`
8. `recovery/restored_from_pyc/core_dump.txt`
9. `recovery/restored_from_pyc/loop_dump.txt`

## Current Breakage Snapshot
- `tests/training/trainer/test_step.py`: failing
- `tests/training/trainer/test_loop.py`: failing
- `tests/training/test_trainer.py`: passing

## Requirement Matrix

### R1: StepExecutor batch contract (3-value return)
- Requirement:
  - `_prepare_batch()` returns `(images, text_embeddings, text_mask)`.
  - No `pooled_text_embed` in this contract.
- Evidence:
  - `tests/training/trainer/test_step.py:292`
  - `tests/training/trainer/test_step.py:303`
  - note in test: pooled text embed removed (`tests/training/trainer/test_step.py:625`)
- Current mismatch:
  - `src/training/trainer/step.py` returns 4 values and carries `pooled_text_embed`.

### R2: cfg_dropout functional behavior
- Requirement:
  - `cfg_dropout==0`: keep embeddings.
  - `cfg_dropout>0` and no `text_encoder`: raise ValueError.
  - `cfg_dropout>0` and no embeddings: raise ValueError.
  - Drop path replaces with null embedding and boolean mask.
- Evidence:
  - `tests/training/trainer/test_step.py:285`
  - `tests/training/trainer/test_step.py:308`
  - `tests/training/trainer/test_step.py:317`
  - docs: `publicdocs/IMPLEMENTATION.md` section 5.9
  - pyc hint: `recovery/restored_from_pyc/step_dump.txt` (`_get_null_text_embed`, `_apply_cfg_dropout`)
- Current mismatch:
  - behavior not matching tests (exceptions and replacement path).

### R3: step_time must be zero-safe
- Requirement:
  - metric computation must avoid division-by-zero for `samples_per_sec`.
- Evidence:
  - failures at `src/training/trainer/step.py:283`
- Current mismatch:
  - direct division by `step_time`.

### R4: Epoch semantics are step-driven
- Requirement:
  - epoch boundary based on `step % len(dataloader) == 0`.
  - not only physical `StopIteration`.
- Evidence:
  - docs: `publicdocs/IMPLEMENTATION.md` lines around section 5.6 step semantics
  - tests: `tests/training/trainer/test_loop.py:167`, `tests/training/trainer/test_loop.py:407`
- Current mismatch:
  - loop epoch increments only on iterator exhaustion path.

### R5: checkpoint_dir fallback when save_path is None
- Requirement:
  - if `save_path` missing, fallback to config `checkpoint_dir` (for periodic saves too).
- Evidence:
  - `tests/training/trainer/test_loop.py:733`
  - `tests/training/trainer/test_loop.py:786`
- Current mismatch:
  - fallback tied to `save_every_epochs > 0` only.

### R6: checkpoint naming format
- Requirement:
  - periodic name format: `checkpoint_epoch{epoch}_step{step}`
  - old `checkpoint_epoch_{N}` must not appear
- Evidence:
  - `tests/training/trainer/test_loop.py:861`
  - docs: `publicdocs/api/API_REFERENCE.md` v1.5.1 notes
  - docs: `publicdocs/architecture/ARCHITECTURE.md` v1.5.1 notes
  - docs: `publicdocs/IMPLEMENTATION.md` section 5.11
- Current mismatch:
  - loop uses `checkpoint_epoch_{epoch}`.

### R7: dedupe periodic checkpoint when step + epoch triggers coincide
- Requirement:
  - if both triggers hit same step, save once.
- Evidence:
  - `tests/training/trainer/test_loop.py:951`
  - docs: `publicdocs/IMPLEMENTATION.md` section 5.11
- Current mismatch:
  - no dedupe logic in current loop implementation.

## Historical Hints From Pyc Recovery
- `step_dump.txt` confirms historical implementations for:
  - `_apply_cfg_dropout`
  - `_get_null_text_embed`
  - `_prepare_batch` returns 3 values
- `loop_dump.txt` confirms historical loop had:
  - more advanced loop orchestration
  - checkpoint name pattern includes `checkpoint_epoch{...}_step{...}`
- Caveat:
  - decompiled output has syntax artifacts; use as behavioral hint, not direct paste source.

## Repair Priority
1. `src/training/trainer/step.py` (R1/R2/R3)
2. `src/training/trainer/loop.py` (R4/R5/R6/R7)
3. Re-run test trio:
   - `tests/training/trainer/test_step.py`
   - `tests/training/trainer/test_loop.py`
   - `tests/training/test_trainer.py`

### R8: Git Workflow Safety (User Directive, 2026-02-11)
- Requirement:
  - Future code edits should be finalized with local git commits after verification.
  - Do not use git rollback-style operations without explicit user approval.
- Explicitly forbidden without user authorization:
  - `git reset --hard`
  - `git checkout -- <path>`
  - `git restore --source ...`
  - `git revert ...` (for rollback purpose)
- Rationale:
  - Prevent accidental destruction of uncommitted local architecture changes.

## Beyond Unit Tests (Non-invasive)
Add tester-only smoke checks under `tests/tester/recovery/`:
- step-time zero-safe metric path
- cfg_dropout null-embedding path
- epoch boundary sync at exact step multiples
- checkpoint naming and dedupe for boundary-coincident steps

These should not modify `src` and should be runnable independently.

## Smoke Status (2026-02-11)
Command:
- `py -3.10 tests\tester\recovery\run_recovery_smoke.py`

Observed failures:
- `_prepare_batch` contract mismatch (`tuple_len=4`)
- `cfg_dropout` missing required error path (no `text_encoder`)
- constant-clock run raises `ZeroDivisionError` in metrics path
- epoch boundary sync mismatch (`epoch=2` at `step=9`, expected 3)
- checkpoint contracts mismatch (old `checkpoint_epoch_1`, missing `_step{N}`)

Observed pass:
- normal-clock `metrics_zero_safe` may pass in some runs (non-deterministic timing), so deterministic constant-clock check is required.
