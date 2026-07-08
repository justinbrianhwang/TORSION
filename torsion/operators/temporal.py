"""Temporal activation patterns for torsion injection."""

from __future__ import annotations

from typing import Literal

TemporalPattern = Literal["single-frame", "burst", "persistent"]


def is_active(
    pattern: str,
    frame_idx: int,
    *,
    start_frame: int,
    duration: int | None = None,
) -> bool:
    """Return whether a temporal fault pattern is active on a frame.

    Supported Phase 1 patterns:
    - single-frame: active only when ``frame_idx == start_frame``.
    - burst: active for ``duration`` frames, start inclusive and end exclusive.
    - persistent: active from ``start_frame`` through the rest of the episode.
    """

    if frame_idx < 0:
        raise ValueError("frame_idx must be non-negative")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")

    normalized = pattern.lower().strip().replace("_", "-")
    if normalized == "single-frame":
        return frame_idx == start_frame
    if normalized == "burst":
        if duration is None or duration <= 0:
            raise ValueError("burst pattern requires a positive duration")
        return start_frame <= frame_idx < start_frame + duration
    if normalized == "persistent":
        return frame_idx >= start_frame
    if normalized in {"drift", "actor-locked"}:
        raise NotImplementedError(f"{pattern!r} is TODO(phase-2/3)")

    raise ValueError(f"unknown temporal pattern {pattern!r}")
