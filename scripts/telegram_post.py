#!/usr/bin/env python3
"""Post a PitchEdge card + caption to Telegram (sample or DB disagreement).

Modes:
  sample       — fixed Spain vs Morocco smoke card (no DB / no Anthropic).
  disagreement — ``post_daily_disagreement`` from logged ``match_predictions``.

Usage:
    make telegram-test              # live sample post
    make telegram-post              # live DB disagreement (needs ANTHROPIC_API_KEY)
    uv run python scripts/telegram_post.py disagreement --dry-run

Loads ``.env`` from the repo root before reading ``pitchedge.config``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _bootstrap_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
    import pitchedge.config as cfg

    importlib.reload(cfg)


def _require_telegram() -> None:
    from pitchedge import config

    if not config.TELEGRAM_BOT_TOKEN.strip():
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set in .env")
    if not config.TELEGRAM_CHAT_ID.strip():
        raise SystemExit("TELEGRAM_CHAT_ID is not set in .env")


def post_sample(*, dry_run: bool) -> None:
    from pitchedge.content.card import CardData, render_card
    from pitchedge.content.telegram import build_caption, post_match_preview

    narrative = {
        "headline": "PitchEdge test — Spain vs Morocco",
        "preview": (
            "Sample card post. Model: Spain 54%, draw 27%, Morocco 19%. "
            "Market row on the card for comparison."
        ),
        "model_read": (
            "Our model is lower on Spain than the de-vigged market on this fixture "
            "(test post only — not a performance claim)."
        ),
        "confidence_note": "Receipts logged before kickoff.",
    }
    card_path = Path(tempfile.mkdtemp(prefix="pitchedge_tg_")) / "sample_card.png"
    render_card(
        CardData(
            home="Spain",
            away="Morocco",
            stage="Round of 16",
            kickoff_local="Jul 5, 15:00 UTC",
            venue="Atlanta",
            p_home=0.54,
            p_draw=0.27,
            p_away=0.19,
            exp_home_goals=1.7,
            exp_away_goals=0.9,
            headline=narrative["headline"],
            market_p_home=0.61,
            market_p_draw=0.24,
            market_p_away=0.15,
        ),
        str(card_path),
    )
    result = post_match_preview(narrative, str(card_path), dry_run=dry_run)
    if dry_run:
        print("dry_run: sample card ready,", card_path)
    else:
        print(f"posted sample message_id={result.get('message_id')}")


def post_disagreement(*, dry_run: bool, db_url: str | None, within_hours: int) -> None:
    from pitchedge import config, db
    from pitchedge.content.daily_disagreement import fetch_candidates
    from pitchedge.content.telegram import post_daily_disagreement

    url = db_url or config.DB_URL
    with db.connect(url) as conn:
        candidates = fetch_candidates(conn, within_hours=within_hours)
    if not candidates:
        raise SystemExit(
            "no upcoming fixtures with paired model+market predictions; run make predict"
        )
    print(f"loaded {len(candidates)} candidate fixtures from DB")
    if not config.ANTHROPIC_API_KEY.strip() and not dry_run:
        raise SystemExit("ANTHROPIC_API_KEY is not set (required for disagreement narrative)")

    # ``post_match_preview`` returns None on dry_run even when assembly succeeds.
    result = post_daily_disagreement(
        candidates,
        min_tvd=0.0,
        dry_run=dry_run,
    )
    if dry_run:
        print("dry_run: disagreement card + caption assembled (not sent)")
        return
    if result is None:
        raise SystemExit("no disagreement selected (empty candidates or threshold)")
    print(f"posted disagreement message_id={result.get('message_id')}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _bootstrap_env()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("sample", "disagreement"),
        help="sample smoke card or DB-backed daily disagreement",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render and log only; do not call Telegram API",
    )
    parser.add_argument(
        "--within-hours",
        type=int,
        default=24 * 365,
        help="disagreement: look ahead window for upcoming fixtures (default: 1 year)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="override DB_URL (default: config / Makefile DOCKER_DB_URL)",
    )
    args = parser.parse_args()

    if not args.dry_run:
        _require_telegram()

    if args.mode == "sample":
        post_sample(dry_run=args.dry_run)
    else:
        post_disagreement(
            dry_run=args.dry_run,
            db_url=args.db_url,
            within_hours=args.within_hours,
        )


if __name__ == "__main__":
    main()
