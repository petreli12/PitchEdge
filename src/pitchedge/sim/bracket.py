"""Round-of-32 bracket construction per FIFA 2026 match schedule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pitchedge.sim.annex_c import SLOTS_FOR_THIRD, resolve_third_place_slots

if TYPE_CHECKING:
    from pitchedge.sim.sampling import MatchupPair

# Fixed R32 pairings (openfootball / FIFA Art. 12.6). Third-place sides use Annex C.
R32_TEMPLATE: tuple[tuple[int, str, str], ...] = (
    (73, "2A", "2B"),
    (74, "1E", "3RD:1E"),
    (75, "1F", "2C"),
    (76, "1C", "2F"),
    (77, "1I", "3RD:1I"),
    (78, "2E", "2I"),
    (79, "1A", "3RD:1A"),
    (80, "1L", "3RD:1L"),
    (81, "1D", "3RD:1D"),
    (82, "1G", "3RD:1G"),
    (83, "2K", "2L"),
    (84, "1H", "2J"),
    (85, "1B", "3RD:1B"),
    (86, "1J", "2H"),
    (87, "1K", "3RD:1K"),
    (88, "2D", "2G"),
)

R16_TEMPLATE: tuple[tuple[int, str, str], ...] = (
    (89, "W74", "W77"),
    (90, "W73", "W75"),
    (91, "W76", "W78"),
    (92, "W79", "W80"),
    (93, "W83", "W84"),
    (94, "W81", "W82"),
    (95, "W86", "W88"),
    (96, "W85", "W87"),
)

QF_TEMPLATE: tuple[tuple[int, str, str], ...] = (
    (97, "W89", "W90"),
    (98, "W93", "W94"),
    (99, "W91", "W92"),
    (100, "W95", "W96"),
)

SF_TEMPLATE: tuple[tuple[int, str, str], ...] = (
    (101, "W97", "W98"),
    (102, "W99", "W100"),
)

FINAL_TEMPLATE: tuple[tuple[int, str, str], ...] = ((104, "W101", "W102"),)


@dataclass(frozen=True)
class Qualifiers:
    """Group-stage qualifiers for knockout draw."""

    winners: dict[str, int]
    runners_up: dict[str, int]
    third_by_group: dict[str, int]
    best_eight_third_groups: list[str]
    third_slots: dict[str, str]


def build_qualifiers(
    ranked_by_group: dict[str, list[int]],
    third_by_group: dict[str, int],
    best_eight_third_groups: list[str],
) -> Qualifiers:
    """Derive ``1X`` / ``2X`` / ``3X`` labels and Annex C third-place slot map."""
    winners = {g: ranked_by_group[g][0] for g in ranked_by_group}
    runners_up = {g: ranked_by_group[g][1] for g in ranked_by_group}
    third_slots = resolve_third_place_slots(best_eight_third_groups)
    return Qualifiers(
        winners=winners,
        runners_up=runners_up,
        third_by_group=third_by_group,
        best_eight_third_groups=best_eight_third_groups,
        third_slots=third_slots,
    )


def _resolve_label(label: str, q: Qualifiers) -> int:
    if label.startswith("3RD:"):
        slot = label.split(":", 1)[1]
        if slot not in SLOTS_FOR_THIRD:
            raise ValueError(f"unknown third-place slot {slot!r}")
        third_label = q.third_slots[slot]
        group = third_label[1]
        return q.third_by_group[group]
    if label[0] in ("1", "2") and len(label) == 2:
        group = label[1]
        if label[0] == "1":
            return q.winners[group]
        return q.runners_up[group]
    raise ValueError(f"cannot resolve label {label!r}")


def simulate_knockout(
    q: Qualifiers,
    model,
    wc_to_model_id: dict[int, int],
    rng,
    *,
    catalog: dict[MatchupPair, object] | None = None,
) -> tuple[int, dict[str, int]]:
    """Simulate R32 → final; return champion ``team_id`` and all ``W*`` winners."""
    from pitchedge.sim.knockout import play_knockout_match

    state: dict[str, int] = {}
    for match_no, home_label, away_label in R32_TEMPLATE:
        home_id = _resolve_label(home_label, q)
        away_id = _resolve_label(away_label, q)
        win_id = play_knockout_match(
            model, home_id, away_id, wc_to_model_id, rng, catalog=catalog
        )
        state[f"W{match_no}"] = win_id

    for template in (R16_TEMPLATE, QF_TEMPLATE, SF_TEMPLATE):
        for match_no, home_label, away_label in template:
            home_id = state[home_label]
            away_id = state[away_label]
            win_id = play_knockout_match(
                model, home_id, away_id, wc_to_model_id, rng, catalog=catalog
            )
            state[f"W{match_no}"] = win_id

    match_no, home_label, away_label = FINAL_TEMPLATE[0]
    home_id = state[home_label]
    away_id = state[away_label]
    champion = play_knockout_match(
        model, home_id, away_id, wc_to_model_id, rng, catalog=catalog
    )
    state[f"W{match_no}"] = champion
    return champion, state
