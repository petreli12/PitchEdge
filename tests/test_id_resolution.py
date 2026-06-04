"""WC team_id -> model id resolution: one shared path for predict.py and the sim.

The disagreement-board bug existed because ``predict.py`` and the tournament sim
had separate id-translation paths and one was broken. These tests pin the two
onto a single resolver and assert every WC team resolves identically and never
falls through to the model's coin-flip default.
"""

from __future__ import annotations

import pytest

import pitchedge.predict as predict_mod
import pitchedge.sim.tournament as tournament_mod
from pitchedge import config
from pitchedge.model.dixon_coles import filter_recency_window, load_training_matches
from pitchedge.sim.wc_teams import (
    assert_wc_teams_in_model,
    load_wc_teams,
    wc_id_to_model_id,
)


def test_predict_and_sim_use_same_resolver_symbol():
    """Both modules must reference the one canonical resolver (cannot diverge)."""
    canonical = wc_id_to_model_id
    assert predict_mod.wc_id_to_model_id is canonical
    assert tournament_mod.wc_id_to_model_id is canonical


def test_all_48_wc_teams_resolve_identically_across_paths():
    teams = load_wc_teams()
    assert len(teams) == 48

    # predict.py path: resolver self-loads. sim path: resolver from loaded teams.
    predict_map = wc_id_to_model_id()
    sim_map = wc_id_to_model_id(teams)

    differing = {
        t.team_id: (predict_map.get(t.team_id), sim_map.get(t.team_id))
        for t in teams
        if predict_map.get(t.team_id) != sim_map.get(t.team_id)
    }
    assert differing == {}, f"teams resolve differently between paths: {differing}"
    assert len(predict_map) == 48


def test_no_two_wc_teams_collide_to_same_model_id():
    """A collision would merge two nations' histories into one model entity."""
    teams = load_wc_teams()
    model_ids = [t.model_team_id for t in teams]
    dupes = {m for m in model_ids if model_ids.count(m) > 1}
    assert dupes == set(), f"WC teams collide onto shared model ids: {dupes}"


def test_assert_wc_teams_in_model_flags_fallthrough():
    mapping = {1: 111, 5: 555}
    assert_wc_teams_in_model(mapping, {111, 555, 999})  # all present -> ok
    with pytest.raises(KeyError):
        assert_wc_teams_in_model(mapping, {111})  # 555 missing -> coin-flip risk


def test_no_wc_team_falls_through_to_model_default(migrated_db):
    """Every WC team must exist in the fitted model's team universe.

    The fit builds its team set from matches inside the recency window, so a WC
    team absent from that set would hit ``attack.get(id, 0.0)`` -> coin flip.
    """
    matches = load_training_matches()
    # Only meaningful against the full ingested dataset; a partial/empty test DB
    # would lack the WC nations' histories and trivially "fall through".
    if len(matches) < 10_000:
        pytest.skip("full raw_results history not loaded; run ingest-history")

    work = filter_recency_window(matches, recency_years=config.DC_RECENCY_YEARS)
    model_universe = {int(r["home_id"]) for r in work} | {
        int(r["away_id"]) for r in work
    }

    wc_to_model = wc_id_to_model_id()
    # Should not raise; surfaces the specific offending teams if it does.
    assert_wc_teams_in_model(wc_to_model, model_universe)
