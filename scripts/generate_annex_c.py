#!/usr/bin/env python3
"""Regenerate ``data/wc2026/annex_c.json`` from FIFA Regulations Annex C (official PDF)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "FWC26_regulations_EN.pdf"

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "verify_annex_c", ROOT / "scripts" / "verify_annex_c.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
extract_annex_text_from_pdf = _mod.extract_annex_text_from_pdf
parse_official_rows = _mod.parse_official_rows


def main() -> None:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    if not pdf_path.is_file():
        print(f"Official PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    raw = extract_annex_text_from_pdf(pdf_path)
    combos = parse_official_rows(raw)
    if len(combos) != 495:
        print(f"expected 495 combinations, parsed {len(combos)}", file=sys.stderr)
        sys.exit(1)

    out_path = ROOT / "data/wc2026/annex_c.json"
    payload = {
        "source": "FIFA World Cup 26 Regulations, Annex C (combinations 1-495)",
        "slots_for_third_place": ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"],
        "combinations": combos,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path} ({len(combos)} combinations)")


if __name__ == "__main__":
    main()
