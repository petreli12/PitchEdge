"""FIFA World Cup 2026 Annex C third-place → Round-of-32 slot mapping.

Source: ``data/wc2026/annex_c.json``, transcribed from FIFA Regulations Annex C
(495 combinations). Each key is the sorted eight group letters whose third-
placed teams qualify; each value maps winner slots ``1A`` … ``1L`` (eight slots
that face a third-placed team) to ``3X`` labels.

Confirm the JSON against the official PDF before production reliance; regenerate
via ``scripts/generate_annex_c.py`` if the regulations are updated.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from pitchedge import config

SLOTS_FOR_THIRD: tuple[str, ...] = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")


def third_place_combination_key(qualifying_group_letters: list[str]) -> str:
    """Return the Annex C lookup key for eight qualifying third-place groups."""
    if len(qualifying_group_letters) != 8:
        raise ValueError(f"expected 8 qualifying third-place groups, got {len(qualifying_group_letters)}")
    return "".join(sorted(qualifying_group_letters))


@lru_cache(maxsize=1)
def load_annex_c(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Load the Annex C combination table from JSON."""
    p = Path(path or config.ANNEX_C_JSON_PATH)
    data = json.loads(p.read_text(encoding="utf-8"))
    combos = data["combinations"]
    if len(combos) != 495:
        raise ValueError(f"annex_c.json must contain 495 combinations, found {len(combos)}")
    return combos


def resolve_third_place_slots(
    qualifying_group_letters: list[str],
    *,
    annex: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, str]:
    """Map winner slots (``1A``, ``1B``, …) to ``3X`` labels for this combination."""
    table = annex if annex is not None else load_annex_c()
    key = third_place_combination_key(qualifying_group_letters)
    try:
        return dict(table[key])
    except KeyError as exc:
        raise KeyError(
            f"no Annex C row for third-place combination {key!r}; "
            "check annex_c.json against FIFA regulations"
        ) from exc
