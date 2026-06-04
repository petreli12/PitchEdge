"""Match and tournament models (probabilities in [0, 1])."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "blend",
    "devig_three_way",
    "elo_win_prob",
    "fit_elo_from_db",
    "fit_dixon_coles",
    "fit_dixon_coles_from_db",
    "match_probs",
    "score_matrix",
    "tournament_match_probs",
    "tournament_score_matrix",
    "wc_match_probs",
    "wc_score_matrix",
]

if TYPE_CHECKING:
    from pitchedge.model.blend import blend
    from pitchedge.model.devig import devig_three_way
    from pitchedge.model.dixon_coles import (
        fit_dixon_coles,
        fit_dixon_coles_from_db,
        match_probs,
        score_matrix,
        tournament_match_probs,
        tournament_score_matrix,
        wc_match_probs,
        wc_score_matrix,
    )
    from pitchedge.model.elo import elo_win_prob, fit_elo_from_db


def __getattr__(name: str):
    if name == "devig_three_way":
        from pitchedge.model.devig import devig_three_way

        return devig_three_way
    if name == "blend":
        from pitchedge.model.blend import blend

        return blend
    if name in ("elo_win_prob", "fit_elo_from_db"):
        from pitchedge.model import elo as elo_mod

        return getattr(elo_mod, name)
    if name in (
        "fit_dixon_coles",
        "fit_dixon_coles_from_db",
        "match_probs",
        "score_matrix",
        "tournament_match_probs",
        "tournament_score_matrix",
        "wc_match_probs",
        "wc_score_matrix",
    ):
        from pitchedge.model import dixon_coles as dc_mod

        return getattr(dc_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
