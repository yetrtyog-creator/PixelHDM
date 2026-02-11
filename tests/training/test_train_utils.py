from __future__ import annotations

from pathlib import Path

from src.training.train_utils import (
    epoch_for_checkpoint_step,
    find_latest_checkpoint,
    repair_checkpoint_filenames,
)


def _touch(path: Path) -> None:
    path.write_bytes(b"ckpt")


def test_epoch_for_checkpoint_step_formula() -> None:
    steps_per_epoch = 9728
    assert epoch_for_checkpoint_step(2500, steps_per_epoch) == 1
    assert epoch_for_checkpoint_step(9728, steps_per_epoch) == 1
    assert epoch_for_checkpoint_step(10000, steps_per_epoch) == 2
    assert epoch_for_checkpoint_step(12500, steps_per_epoch) == 2


def test_repair_checkpoint_filenames_renames_misnamed_file(tmp_path: Path) -> None:
    wrong = tmp_path / "checkpoint_epoch3_step12500.pt"
    _touch(wrong)

    renamed = repair_checkpoint_filenames(tmp_path, steps_per_epoch=9728, dry_run=False)

    assert renamed == 1
    assert not wrong.exists()
    assert (tmp_path / "checkpoint_epoch2_step12500.pt").exists()


def test_repair_checkpoint_filenames_keeps_canonical_name(tmp_path: Path) -> None:
    canonical = tmp_path / "checkpoint_epoch2_step12500.pt"
    _touch(canonical)

    renamed = repair_checkpoint_filenames(tmp_path, steps_per_epoch=9728, dry_run=False)

    assert renamed == 0
    assert canonical.exists()


def test_repair_checkpoint_filenames_skips_when_target_exists(tmp_path: Path) -> None:
    wrong = tmp_path / "checkpoint_epoch3_step12500.pt"
    target = tmp_path / "checkpoint_epoch2_step12500.pt"
    _touch(wrong)
    _touch(target)

    renamed = repair_checkpoint_filenames(tmp_path, steps_per_epoch=9728, dry_run=False)

    assert renamed == 0
    assert wrong.exists()
    assert target.exists()


def test_repair_checkpoint_filenames_falls_back_to_copy_on_rename_error(
    tmp_path: Path, monkeypatch
) -> None:
    wrong = tmp_path / "checkpoint_epoch3_step12500.pt"
    _touch(wrong)

    def _raise_rename(self, target):
        raise OSError("locked")

    monkeypatch.setattr(Path, "rename", _raise_rename)

    renamed = repair_checkpoint_filenames(tmp_path, steps_per_epoch=9728, dry_run=False)

    assert renamed == 1
    assert wrong.exists()
    assert (tmp_path / "checkpoint_epoch2_step12500.pt").exists()


def test_find_latest_checkpoint_prefers_lower_epoch_on_same_step_tie(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "checkpoint_epoch3_step12500.pt"
    canonical = tmp_path / "checkpoint_epoch2_step12500.pt"
    _touch(wrong)
    _touch(canonical)

    # Force same mtime to exercise tie-break behavior.
    mtime = 1_700_000_000
    wrong.touch()
    canonical.touch()
    import os
    os.utime(wrong, (mtime, mtime))
    os.utime(canonical, (mtime, mtime))

    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint_epoch2_step12500.pt"
