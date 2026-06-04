#!/usr/bin/env python3
"""Compare ``data/wc2026/annex_c.json`` to FIFA Regulations Annex C (official PDF)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "FWC26_regulations_EN.pdf"
JSON_PATH = ROOT / "data/wc2026/annex_c.json"
SLOTS = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")
ANNEX_PAGES = range(79, 97)  # PDF pages 80–97 (0-based indices)
ROW_RE = re.compile(r"^\s*(\d{1,3})\s+((?:3[A-L]\s+){7}3[A-L])\s*$")


def extract_annex_text_from_pdf(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "\n".join(reader.pages[i].extract_text() or "" for i in ANNEX_PAGES)


def parse_official_rows(raw: str) -> dict[str, dict[str, str]]:
    """Parse options 1–495 from Annex C table lines in the regulations PDF."""
    entries: dict[str, dict[str, str]] = {}
    for line in raw.splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        opt = int(m.group(1))
        if opt < 1 or opt > 495:
            continue
        groups = [tok[1] for tok in m.group(2).split()]
        key = "".join(sorted(groups))
        entries[key] = {slot: f"3{g}" for slot, g in zip(SLOTS, groups, strict=True)}
    return entries


def load_json_combos() -> dict[str, dict[str, str]]:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return data["combinations"]


def compare(
    official: dict[str, dict[str, str]],
    stored: dict[str, dict[str, str]],
) -> list[str]:
    errors: list[str] = []
    if len(official) != 495:
        errors.append(f"official parse count={len(official)} (expected 495)")
    if len(stored) != 495:
        errors.append(f"json count={len(stored)} (expected 495)")

    for key in sorted(set(official) | set(stored)):
        if key not in official:
            errors.append(f"key {key!r}: in json only")
            continue
        if key not in stored:
            errors.append(f"key {key!r}: in official PDF only")
            continue
        for slot in SLOTS:
            o = official[key].get(slot)
            s = stored[key].get(slot)
            if o != s:
                errors.append(
                    f"key={key} slot={slot}: json={s!r} official={o!r}"
                )
    return errors


def main() -> None:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    if not pdf_path.is_file():
        print(f"Official PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(2)

    raw = extract_annex_text_from_pdf(pdf_path)
    official = parse_official_rows(raw)
    stored = load_json_combos()
    errors = compare(official, stored)

    if errors:
        print(f"MISMATCH: {len(errors)} cell/key differences")
        for line in errors[:50]:
            print(line)
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more")
        sys.exit(1)

    print(
        f"OK: all 495 Annex C combinations match "
        f"{pdf_path.name} (pages 80–97)"
    )


if __name__ == "__main__":
    main()
