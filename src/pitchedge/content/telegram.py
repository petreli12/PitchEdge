"""
telegram.py — distribution layer. Posts a caption + card image to the PitchEdge
Telegram channel, and assembles the full daily-disagreement post by tying
together daily_disagreement -> narrative -> card -> send.

All HTTP goes through a single injectable chokepoint (`_api_call`) so unit tests
run with a mock session and never touch the network. Every post function accepts
`dry_run=True`, which logs the assembled payload and returns without sending —
this is what `scheduler.py --dry-run` uses.

Telegram limits respected: photo caption <= 1024 chars (we truncate safely).
Parse mode is HTML (simpler/safer than MarkdownV2); user text is escaped.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional, Sequence

import requests

from pitchedge import config  # TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from pitchedge.content.narrative import NarrativeInput, generate_narrative
from pitchedge.content.card import CardData, render_card
from pitchedge.content.daily_disagreement import Candidate, select_top, to_narrative_input

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org/bot{token}/{method}"
CAPTION_LIMIT = 1024
_HANDLE = config.TELEGRAM_CHANNEL_HANDLE  # footer attribution; from config/env


# --------------------------------------------------------------------------- #
# HTTP chokepoint                                                             #
# --------------------------------------------------------------------------- #
def _api_call(method: str, *, data: dict, files: Optional[dict] = None,
              session: Optional[requests.Session] = None) -> dict:
    """Single point of contact with the Telegram Bot API. Injectable session."""
    session = session or requests
    url = API_BASE.format(token=config.TELEGRAM_BOT_TOKEN, method=method)
    resp = session.post(url, data=data, files=files, timeout=30)
    payload = resp.json()
    if not payload.get("ok"):
        raise RuntimeError(
            f"Telegram {method} failed: {payload.get('description', resp.text)}"
        )
    return payload["result"]


# --------------------------------------------------------------------------- #
# Caption assembly                                                            #
# --------------------------------------------------------------------------- #
def _esc(s: str) -> str:
    """Escape for Telegram HTML parse mode."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_caption(narrative: dict) -> str:
    """Assemble a caption from the narrative dict (headline/preview/model_read/
    confidence_note). Order: bold headline, body, optional model read, optional
    confidence note, footer. Truncated to the Telegram caption limit."""
    parts = [f"<b>{_esc(narrative['headline'])}</b>", "", _esc(narrative["preview"])]
    if narrative.get("model_read"):
        parts += ["", _esc(narrative["model_read"])]
    if narrative.get("confidence_note"):
        parts += ["", f"<i>{_esc(narrative['confidence_note'])}</i>"]
    parts += ["", f"Logged before kickoff · {_HANDLE}"]
    caption = "\n".join(parts)

    if len(caption) > CAPTION_LIMIT:
        # Trim the body first, preserving headline + footer.
        footer = f"\n\nLogged before kickoff · {_HANDLE}"
        head = f"<b>{_esc(narrative['headline'])}</b>\n\n"
        room = CAPTION_LIMIT - len(head) - len(footer) - 1
        body = _esc(narrative["preview"])[: max(room, 0)].rstrip()
        caption = head + body + "…" + footer
    return caption


# --------------------------------------------------------------------------- #
# Posting                                                                     #
# --------------------------------------------------------------------------- #
def post_photo(image_path: str, caption: str, *,
               session: Optional[requests.Session] = None,
               dry_run: bool = False) -> Optional[dict]:
    """Send a photo with caption to the configured channel."""
    if dry_run:
        log.info("[dry_run] would post photo %s with caption:\n%s", image_path, caption)
        return None
    with open(image_path, "rb") as fh:
        return _api_call(
            "sendPhoto",
            data={"chat_id": config.TELEGRAM_CHAT_ID, "caption": caption,
                  "parse_mode": "HTML"},
            files={"photo": fh},
            session=session,
        )


def post_match_preview(narrative: dict, card_path: str, *,
                       session: Optional[requests.Session] = None,
                       dry_run: bool = False) -> Optional[dict]:
    """Post a single fixture preview: card image + assembled caption."""
    return post_photo(card_path, build_caption(narrative),
                      session=session, dry_run=dry_run)


# --------------------------------------------------------------------------- #
# Orchestration: the full daily-disagreement post                            #
# --------------------------------------------------------------------------- #
def _card_from_narrative(ni: NarrativeInput, headline: str) -> CardData:
    return CardData(
        home=ni.home, away=ni.away, stage=ni.stage, kickoff_local=ni.kickoff_local,
        venue=ni.venue, p_home=ni.p_home, p_draw=ni.p_draw, p_away=ni.p_away,
        exp_home_goals=ni.exp_home_goals, exp_away_goals=ni.exp_away_goals,
        headline=headline,
        market_p_home=ni.market_p_home, market_p_draw=ni.market_p_draw,
        market_p_away=ni.market_p_away,
    )


def post_daily_disagreement(
    candidates: Sequence[Candidate], *,
    narrative_client=None,
    session: Optional[requests.Session] = None,
    min_tvd: Optional[float] = None,
    tmp_dir: Optional[str] = None,
    dry_run: bool = False,
) -> Optional[dict]:
    """End-to-end daily post: pick the biggest model-vs-market divergence, write
    the preview, render the card, and post it. Returns None (and posts nothing)
    if no fixture clears the disagreement threshold."""
    top = select_top(candidates, min_tvd=min_tvd)
    if top is None:
        log.info("no disagreement cleared the bar today; nothing posted")
        return None

    ni = to_narrative_input(top)
    narrative = generate_narrative(ni, client=narrative_client)
    card_data = _card_from_narrative(ni, narrative["headline"])

    tmp_dir = tmp_dir or tempfile.gettempdir()
    card_path = os.path.join(tmp_dir, f"disagreement_{top.candidate.fixture_id}.png")
    render_card(card_data, card_path)

    log.info("daily disagreement: %s vs %s (tvd=%.3f) %s",
             top.candidate.home, top.candidate.away, top.tvd, top.note)
    return post_match_preview(narrative, card_path, session=session, dry_run=dry_run)
