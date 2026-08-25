"""安定なID生成のための小さなユーティリティ。"""

from __future__ import annotations

import re


def make_id(*parts: object) -> str:
    safe = [re.sub(r"\s+", "_", str(p).strip()) for p in parts if p is not None]
    return "::".join(safe)


_ZONE_SUFFIXES = ("基本性能", "性能")


def strip_zone_suffix(label: str) -> str:
    text = label.strip()
    for suffix in _ZONE_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)].strip() or text
    return text
