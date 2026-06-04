"""
narrative.py — generate calibrated, non-hype match previews from structured
model output via the Anthropic API.

Design contract (see docs/PRD.md, .cursorrules):
  * The JSON payload passed to the model is the ONLY source of truth. The model
    is forbidden from inventing form, injuries, head-to-head, lineups, quotes,
    or any number not present in the payload.
  * Probabilities are described honestly. A 42% favorite is "narrowly favored,"
    never "will win." Coin-flips are called coin-flips.
  * Never claim or imply the model beats the market or offers a betting edge.
    This is analysis, not tips. No call to wager.
  * Output is STRICT JSON with a fixed shape so card.py / telegram.py can rely
    on it. No prose, no markdown fences outside the JSON.

Tested with a mocked client (Phase 6 validation) — no live call in unit tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import anthropic

from pitchedge import config  # expects: ANTHROPIC_API_KEY, NARRATIVE_MODEL

log = logging.getLogger(__name__)

# Default model: Sonnet is a good quality/cost point for per-match narrative.
# Switch to a Haiku-class model in config to cut cost if volume bites.
DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 700
TEMPERATURE = 0.4  # low: we want sober, consistent copy, not creativity


# --------------------------------------------------------------------------- #
# Input contract                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class NarrativeInput:
    """Everything the model is allowed to know. If a field isn't here, it does
    not exist as far as the narrative is concerned."""

    home: str
    away: str
    stage: str                 # e.g. "Group H, Matchday 1"
    kickoff_local: str         # human string, already localized at the edge
    venue: Optional[str]       # or None
    # Standalone model probabilities (published; may be temperature-scaled) — ~1.0
    p_home: float
    p_draw: float
    p_away: float
    exp_home_goals: float
    exp_away_goals: float
    # Optional context, all model-derived (never scraped narrative):
    over25_prob: Optional[float] = None
    btts_prob: Optional[float] = None
    home_elo: Optional[int] = None
    away_elo: Optional[int] = None
    # Market comparison (set only when a meaningful disagreement exists):
    market_p_home: Optional[float] = None
    market_p_draw: Optional[float] = None
    market_p_away: Optional[float] = None
    disagreement_note: Optional[str] = None  # e.g. "model +9pts on away vs market"


# --------------------------------------------------------------------------- #
# Output contract                                                             #
# --------------------------------------------------------------------------- #
# {
#   "headline":         str  (<= ~70 chars, factual, no hype)
#   "preview":          str  (2-3 sentences, <= ~80 words)
#   "model_read":       str | null  (one neutral sentence on model vs market)
#   "confidence_note":  str | null  (humility line when the match is close)
# }
REQUIRED_KEYS = {"headline", "preview", "model_read", "confidence_note"}


# --------------------------------------------------------------------------- #
# The prompt                                                                  #
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You write match previews for PitchEdge, a World Cup analytics product whose
entire reputation rests on being calibrated and honest. You are not a tipster.

ABSOLUTE RULES — violating any of these breaks the product:

1. SOURCE OF TRUTH. The JSON the user provides is the ONLY information you have.
   Do not invent or imply form, recent results, injuries, suspensions, lineups,
   head-to-head history, manager quotes, weather, or any statistic that is not
   in the payload. If you weren't given it, it does not exist. Do not round
   numbers in a way that changes their meaning.

2. PROBABILITY DISCIPLINE. Describe the probabilities exactly as honest odds,
   not destiny. Calibrate your language to the numbers:
     - A side under ~45% is "narrowly favored" / "slight edge", never "will win".
     - 45-60% is "clearly favored". Above ~65% is "strong favorite".
     - When the three outcomes are bunched (no outcome far ahead), say so
       plainly: this is a coin-flip / wide-open match. Treat a high draw
       probability as the real possibility it is.
   Never state or imply any result is certain.

3. NO BETTING, NO EDGE CLAIMS. Never tell anyone to bet, never use tipster
   language ("lock", "banker", "easy money", "value", "sure thing"), and NEVER
   claim or imply our model beats the market or has an edge. We present analysis;
   readers decide.

4. MODEL VS MARKET. If market probabilities and a disagreement note are present,
   you may neutrally observe where our model differs from the market — framed as
   an observation, not a recommendation ("our model is more bullish on X than the
   market; the market may be pricing in something our model can't see"). If no
   market data is provided, set model_read to null. Never spin a disagreement as
   us being right.

TONE. Sober, sharp, literate football writing. Confident about the analysis,
humble about the uncertainty. No exclamation marks. No emojis. No hype. No
cliches. British-neutral register is fine. Name the teams; don't pad.

OUTPUT. Respond with a SINGLE JSON object and nothing else — no markdown fences,
no text before or after. Exact keys:
  "headline": string, <= ~70 chars, factual, no hype.
  "preview": string, 2-3 sentences, <= ~80 words, using only payload facts.
  "model_read": string or null — one neutral sentence on model vs market, or
                null if no meaningful market comparison was provided.
  "confidence_note": string or null — one short humility line when the match is
                close or low-confidence (e.g. bunched probabilities), else null.
"""


def build_user_message(data: NarrativeInput) -> str:
    """Serialize the input as the JSON payload the model reasons over. We hand
    it clean JSON rather than prose so there's nothing to 'read between'."""
    payload = {k: v for k, v in asdict(data).items() if v is not None}
    return (
        "Write the preview for this fixture. Use ONLY these facts:\n\n"
        + json.dumps(payload, indent=2)
    )


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #
def _parse_strict_json(text: str) -> dict:
    """Parse model output as JSON, tolerating accidental code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # drop a leading 'json' language tag if present
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:]
    obj = json.loads(cleaned)
    missing = REQUIRED_KEYS - obj.keys()
    if missing:
        raise ValueError(f"narrative output missing keys: {missing}")
    return obj


def _validate_inputs(data: NarrativeInput) -> None:
    s = data.p_home + data.p_draw + data.p_away
    if not (0.98 <= s <= 1.02):
        raise ValueError(f"model probabilities must sum to ~1.0, got {s:.3f}")


def generate_narrative(
    data: NarrativeInput,
    client: Optional[anthropic.Anthropic] = None,
    model: Optional[str] = None,
) -> dict:
    """Return a validated narrative dict for one fixture.

    `client` is injectable so unit tests can pass a mock (no live API call).
    """
    _validate_inputs(data)
    client = client or anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    model = model or getattr(config, "NARRATIVE_MODEL", DEFAULT_MODEL)

    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(data)}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")

    try:
        result = _parse_strict_json(text)
    except (json.JSONDecodeError, ValueError) as e:
        log.error("narrative parse failed (%s); falling back to template", e)
        result = _fallback(data)
    return result


def _fallback(data: NarrativeInput) -> dict:
    """Deterministic, hype-free template used if the model returns bad JSON, so
    the pipeline never blocks on a content failure."""
    fav, fav_p = max(
        ((data.home, data.p_home), (data.away, data.p_away)), key=lambda t: t[1]
    )
    descriptor = (
        "a coin-flip" if abs(data.p_home - data.p_away) < 0.08
        else f"{fav} narrowly favored" if fav_p < 0.45
        else f"{fav} clearly favored"
    )
    return {
        "headline": f"{data.home} vs {data.away}: {descriptor}",
        "preview": (
            f"{data.stage}. Our model gives {data.home} {data.p_home:.0%}, "
            f"the draw {data.p_draw:.0%}, and {data.away} {data.p_away:.0%}, "
            f"with an expected scoreline near "
            f"{data.exp_home_goals:.1f}-{data.exp_away_goals:.1f}."
        ),
        "model_read": None,
        "confidence_note": (
            "Outcomes are bunched here; treat this as genuinely open."
            if abs(data.p_home - data.p_away) < 0.08 else None
        ),
    }


# --------------------------------------------------------------------------- #
# Manual smoke test:  python -m pitchedge.content.narrative                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    sample = NarrativeInput(
        home="Spain", away="Morocco", stage="Round of 16",
        kickoff_local="July 5, 3:00 PM ET", venue="Mercedes-Benz Stadium, Atlanta",
        p_home=0.54, p_draw=0.27, p_away=0.19,
        exp_home_goals=1.7, exp_away_goals=0.9,
        over25_prob=0.46, btts_prob=0.41, home_elo=2080, away_elo=1840,
        market_p_home=0.61, market_p_draw=0.24, market_p_away=0.15,
        disagreement_note="model is 7pts lower on Spain than the market",
    )
    print(json.dumps(generate_narrative(sample), indent=2))
