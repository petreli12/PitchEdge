"""
card.py — render a fixed-template, on-brand PNG for one fixture.

A template, NOT AI image generation: free, instant, visually consistent across
every post. Pillow only. Colors and layout are brand tokens defined once here so
every card is identical in structure and only the data changes.

Layout (1080x1080 square — renders well on Telegram and X):

  ┌──────────────────────────────────────────────────────┐
  │  PITCHEDGE                       calibrated · receipts │   header
  │                                                        │
  │  ROUND OF 16 · Jul 5, 3:00 PM ET · Atlanta             │   context
  │                                                        │
  │      SPAIN            vs            MOROCCO             │   matchup
  │                                                        │
  │   [optional headline line]                             │
  │                                                        │
  │  ███████████████░░░░░░░░░░░░░░░░░░░  (stacked bar)     │   prob bar
  │  Spain 54%        Draw 27%       Morocco 19%           │   legend
  │                                                        │
  │  Expected score  1.7 – 0.9                             │   xg
  │  Market          61% · 24% · 15%   (optional row)      │   market
  │                                                        │
  │  Probabilities logged before kickoff.                 │   footer
  └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# Brand tokens                                                                #
# --------------------------------------------------------------------------- #
W = H = 1080
PAD = 72

BG = (14, 17, 22)            # near-black
SURFACE = (22, 27, 34)       # panel
TEXT = (240, 243, 246)       # primary
MUTED = (139, 148, 158)      # secondary
HOME_C = (76, 141, 255)      # blue
DRAW_C = (110, 118, 129)     # grey
AWAY_C = (255, 107, 92)      # coral
ACCENT = (63, 185, 80)       # green — the "receipts" brand color

# Font paths: DejaVu ships with most Linux/Pillow installs. Override via args if
# you drop a brand font into the repo. Falls back gracefully.
FONT_BOLD = "DejaVuSans-Bold.ttf"
FONT_REG = "DejaVuSans.ttf"


@dataclass
class CardData:
    home: str
    away: str
    stage: str
    kickoff_local: str
    venue: Optional[str]
    p_home: float
    p_draw: float
    p_away: float
    exp_home_goals: float
    exp_away_goals: float
    headline: Optional[str] = None
    # optional market comparison row
    market_p_home: Optional[float] = None
    market_p_draw: Optional[float] = None
    market_p_away: Optional[float] = None


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _text_w(draw: ImageDraw.ImageDraw, s: str, font) -> int:
    return draw.textbbox((0, 0), s, font=font)[2]


def render_card(data: CardData, out_path: str,
                font_bold: str = FONT_BOLD, font_reg: str = FONT_REG) -> str:
    """Render the fixture card to `out_path` (PNG). Returns the path."""
    assert abs(data.p_home + data.p_draw + data.p_away - 1.0) < 0.02, \
        "probabilities must sum to ~1.0"

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    f_brand = _font(font_bold, 40)
    f_tag = _font(font_reg, 26)
    f_ctx = _font(font_reg, 30)
    f_team = _font(font_bold, 64)
    f_vs = _font(font_reg, 36)
    f_head = _font(font_reg, 34)
    f_legend = _font(font_bold, 32)
    f_label = _font(font_reg, 30)
    f_foot = _font(font_reg, 26)

    # Header
    d.text((PAD, PAD), "PITCHEDGE", font=f_brand, fill=TEXT)
    tag = "calibrated · shows its receipts"
    d.text((W - PAD - _text_w(d, tag, f_tag), PAD + 10), tag, font=f_tag, fill=ACCENT)

    # Context line
    ctx = f"{data.stage.upper()} · {data.kickoff_local}"
    if data.venue:
        ctx += f" · {data.venue}"
    d.text((PAD, PAD + 90), ctx, font=f_ctx, fill=MUTED)

    # Matchup
    y_team = 300
    d.text((PAD, y_team), data.home.upper(), font=f_team, fill=TEXT)
    vs = "vs"
    d.text(((W - _text_w(d, vs, f_vs)) // 2, y_team + 18), vs, font=f_vs, fill=MUTED)
    away_up = data.away.upper()
    d.text((W - PAD - _text_w(d, away_up, f_team), y_team), away_up,
           font=f_team, fill=TEXT)

    # Optional headline
    y = y_team + 120
    if data.headline:
        d.text((PAD, y), data.headline, font=f_head, fill=TEXT)
        y += 70

    # Stacked probability bar
    bar_y, bar_h = y + 40, 56
    bar_x0, bar_x1 = PAD, W - PAD
    bar_w = bar_x1 - bar_x0
    segs = [(data.p_home, HOME_C), (data.p_draw, DRAW_C), (data.p_away, AWAY_C)]
    x = bar_x0
    for frac, color in segs:
        seg_w = bar_w * frac
        d.rectangle([x, bar_y, x + seg_w, bar_y + bar_h], fill=color)
        x += seg_w

    # Legend
    leg_y = bar_y + bar_h + 28
    home_lbl = f"{data.home} {data.p_home:.0%}"
    draw_lbl = f"Draw {data.p_draw:.0%}"
    away_lbl = f"{data.away} {data.p_away:.0%}"
    d.text((bar_x0, leg_y), home_lbl, font=f_legend, fill=HOME_C)
    d.text(((W - _text_w(d, draw_lbl, f_legend)) // 2, leg_y), draw_lbl,
           font=f_legend, fill=MUTED)
    d.text((bar_x1 - _text_w(d, away_lbl, f_legend), leg_y), away_lbl,
           font=f_legend, fill=AWAY_C)

    # Expected score
    y2 = leg_y + 90
    d.text((PAD, y2), "Expected score", font=f_label, fill=MUTED)
    xg = f"{data.exp_home_goals:.1f} - {data.exp_away_goals:.1f}"
    d.text((W - PAD - _text_w(d, xg, f_legend), y2 - 2), xg, font=f_legend, fill=TEXT)

    # Optional market row
    if None not in (data.market_p_home, data.market_p_draw, data.market_p_away):
        y3 = y2 + 56
        d.text((PAD, y3), "Market", font=f_label, fill=MUTED)
        mkt = (f"{data.market_p_home:.0%} · {data.market_p_draw:.0%} · "
               f"{data.market_p_away:.0%}")
        d.text((W - PAD - _text_w(d, mkt, f_label), y3), mkt, font=f_label, fill=MUTED)

    # Footer (the receipts brand)
    foot = "Probabilities logged before kickoff."
    d.text((PAD, H - PAD - 30), foot, font=f_foot, fill=ACCENT)

    img.save(out_path, "PNG")
    return out_path


# Smoke test:  python -m pitchedge.content.card
if __name__ == "__main__":
    sample = CardData(
        home="Spain", away="Morocco", stage="Round of 16",
        kickoff_local="Jul 5, 3:00 PM ET", venue="Atlanta",
        p_home=0.54, p_draw=0.27, p_away=0.19,
        exp_home_goals=1.7, exp_away_goals=0.9,
        headline="Spain favored, but the market is higher still",
        market_p_home=0.61, market_p_draw=0.24, market_p_away=0.15,
    )
    print("wrote", render_card(sample, "card_sample.png"))
