"""WC ``history_name`` maps display names to Kaggle labels for model ids."""

from __future__ import annotations

from pitchedge import config
from pitchedge.ingest.team_ids import history_label_for_model, team_name_to_id
from pitchedge.sim.wc_teams import load_wc_teams


def test_history_label_for_model_prefers_history_name():
    assert history_label_for_model("Czechia", "Czech Republic") == "Czech Republic"
    assert history_label_for_model("Spain", None) == "Spain"
    assert history_label_for_model("Spain", "") == "Spain"


def test_wc_teams_model_id_uses_history_name_for_mismatched_labels():
    teams = load_wc_teams(config.TEAMS_CSV_PATH)
    by_name = {t.name: t for t in teams}
    cz = by_name["Czechia"]
    assert cz.history_name == "Czech Republic"
    assert cz.model_team_id == team_name_to_id("Czech Republic")
    assert cz.model_team_id != team_name_to_id("Czechia")
    tr = by_name["Turkiye"]
    assert tr.model_team_id == team_name_to_id("Turkey")
    cu = by_name["Curacao"]
    assert cu.model_team_id == team_name_to_id("Curaçao")


def test_spain_unchanged_without_history_name():
    teams = load_wc_teams(config.TEAMS_CSV_PATH)
    spain = next(t for t in teams if t.name == "Spain")
    assert spain.history_name is None
    assert spain.model_team_id == team_name_to_id("Spain")
