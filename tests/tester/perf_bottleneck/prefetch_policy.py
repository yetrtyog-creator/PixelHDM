"""Prefetch policy helpers for runtime profiling runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PrefetchSettings:
    enabled: bool
    multiplier: int
    requested_items: Optional[int]
    resolved_items: int


def resolve_prefetch_settings(
    *,
    gradient_accumulation_steps: int,
    thread_prefetch_enabled: bool,
    prefetch_buffer_multiplier: int,
    prefetch_buffer_items: Optional[int],
) -> PrefetchSettings:
    grad_accum = max(1, int(gradient_accumulation_steps))
    multiplier = max(1, int(prefetch_buffer_multiplier))
    requested_items = None if prefetch_buffer_items is None else max(1, int(prefetch_buffer_items))

    if not thread_prefetch_enabled:
        return PrefetchSettings(
            enabled=False,
            multiplier=multiplier,
            requested_items=requested_items,
            resolved_items=0,
        )

    resolved_items = requested_items if requested_items is not None else (grad_accum * multiplier)
    return PrefetchSettings(
        enabled=True,
        multiplier=multiplier,
        requested_items=requested_items,
        resolved_items=max(1, int(resolved_items)),
    )


def apply_prefetch_metadata(metadata: Dict[str, Any], settings: PrefetchSettings) -> None:
    metadata["thread_prefetch_enabled"] = bool(settings.enabled)
    metadata["prefetch_buffer_multiplier"] = int(settings.multiplier)
    metadata["prefetch_buffer_items"] = settings.requested_items
    metadata["prefetch_resolved_items"] = int(settings.resolved_items)

