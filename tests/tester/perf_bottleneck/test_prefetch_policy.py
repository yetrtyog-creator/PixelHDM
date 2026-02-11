from __future__ import annotations

from tests.tester.perf_bottleneck.prefetch_policy import (
    PrefetchSettings,
    apply_prefetch_metadata,
    resolve_prefetch_settings,
)


def test_default_prefetch_enabled_with_2x_multiplier() -> None:
    settings = resolve_prefetch_settings(
        gradient_accumulation_steps=3,
        thread_prefetch_enabled=True,
        prefetch_buffer_multiplier=2,
        prefetch_buffer_items=None,
    )
    assert settings.enabled is True
    assert settings.multiplier == 2
    assert settings.requested_items is None
    assert settings.resolved_items == 6


def test_disable_prefetch_forces_resolved_items_zero() -> None:
    settings = resolve_prefetch_settings(
        gradient_accumulation_steps=3,
        thread_prefetch_enabled=False,
        prefetch_buffer_multiplier=2,
        prefetch_buffer_items=None,
    )
    assert settings.enabled is False
    assert settings.resolved_items == 0


def test_explicit_buffer_items_overrides_multiplier() -> None:
    settings = resolve_prefetch_settings(
        gradient_accumulation_steps=8,
        thread_prefetch_enabled=True,
        prefetch_buffer_multiplier=2,
        prefetch_buffer_items=5,
    )
    assert settings.enabled is True
    assert settings.resolved_items == 5


def test_multiplier_is_clamped_to_minimum_one() -> None:
    settings = resolve_prefetch_settings(
        gradient_accumulation_steps=4,
        thread_prefetch_enabled=True,
        prefetch_buffer_multiplier=0,
        prefetch_buffer_items=None,
    )
    assert settings.multiplier == 1
    assert settings.resolved_items == 4


def test_apply_prefetch_metadata_writes_contract_fields() -> None:
    metadata = {"run_id": "x"}
    settings = PrefetchSettings(
        enabled=True,
        multiplier=2,
        requested_items=None,
        resolved_items=8,
    )
    apply_prefetch_metadata(metadata, settings)

    assert metadata["thread_prefetch_enabled"] is True
    assert metadata["prefetch_buffer_multiplier"] == 2
    assert metadata["prefetch_buffer_items"] is None
    assert metadata["prefetch_resolved_items"] == 8
