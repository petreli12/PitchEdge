#!/usr/bin/env python3
"""Bracket-path diagnostics for France / England / Portugal title-vs-final gap."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchedge import config
from pitchedge.ingest.history import frame_to_rows, load_history_frame
from pitchedge.model.dixon_coles import (
    fit_dixon_coles,
    fit_dixon_coles_from_db,
    load_latest_elo,
    load_training_matches,
)
from pitchedge.model.elo import fit_elo
from pitchedge.sim.bracket import (
    FINAL_TEMPLATE,
    QF_TEMPLATE,
    R16_TEMPLATE,
    R32_TEMPLATE,
    SF_TEMPLATE,
    build_qualifiers,
    simulate_knockout,
    _resolve_label,
)
from pitchedge.sim.group_batch import (
    accumulate_standings,
    idx_to_team_id_map,
    prepare_group_batch,
    resolve_one_group_outcome,
    sample_all_group_scorelines,
)
from pitchedge.sim.wc_teams import (
    load_group_fixtures,
    load_wc_teams,
    teams_by_group,
    wc_id_to_model_id,
)

# Focus teams (WC team_id from teams.csv)
FRANCE = 33
ENGLAND = 45
PORTUGAL = 41
ARGENTINA = 37
SPAIN = 29
FOCUS = (FRANCE, ENGLAND, PORTUGAL, ARGENTINA, SPAIN)

NAMES = {
    FRANCE: "France",
    ENGLAND: "England",
    PORTUGAL: "Portugal",
    ARGENTINA: "Argentina",
    SPAIN: "Spain",
}


def _build_bracket_maps() -> tuple[dict[int, str], dict[str, str]]:
    """Map each R32 match number to QF bracket id (97–100)."""
    # Propagate placeholder winners up the tree.
    r32_winners = {f"W{m}": f"W{m}" for m, _, _ in R32_TEMPLATE}
    r16_winners: dict[str, str] = {}
    for m, h, a in R16_TEMPLATE:
        r16_winners[f"W{m}"] = f"W{m}"
    qf_winners: dict[str, str] = {}
    for m, h, a in QF_TEMPLATE:
        qf_winners[f"W{m}"] = f"W{m}"

    r32_to_r16: dict[int, int] = {}
    for m, h, a in R16_TEMPLATE:
        for label in (h, a):
            if label.startswith("W") and label[1:].isdigit():
                r32_to_r16[int(label[1:])] = m

    r16_to_qf: dict[int, int] = {}
    for m, h, a in QF_TEMPLATE:
        for label in (h, a):
            if label.startswith("W") and label[1:].isdigit():
                r16_to_qf[int(label[1:])] = m

    r32_to_qf: dict[int, str] = {}
    for r32_m, r16_m in r32_to_r16.items():
        qf_m = r16_to_qf.get(r16_m)
        if qf_m is not None:
            r32_to_qf[r32_m] = f"QF{qf_m}"

    # SF half: 101 = QF97+QF98 side, 102 = QF99+QF100 side
    sf_half: dict[str, str] = {}
    for m, h, a in QF_TEMPLATE:
        slot = f"W{m}"
        if m in (97, 98):
            sf_half[slot] = "SF101_side"
        else:
            sf_half[slot] = "SF102_side"
    return r32_to_qf, sf_half


R32_TO_QF, QF_TO_SF_SIDE = _build_bracket_maps()


def r32_entries_for_team(team_id: int) -> list[tuple[int, str, str]]:
    """All (match_no, slot_label, home|away) slots this team_id can enter from group labels."""
    entries: list[tuple[int, str, str]] = []
    teams = load_wc_teams()
    group = next(t.group_label for t in teams if t.team_id == team_id)
    for m, h, a in R32_TEMPLATE:
        if h in (f"1{group}", f"2{group}") or h == f"3RD:1{group}":
            entries.append((m, h, "home"))
        if a in (f"1{group}", f"2{group}") or a == f"3RD:1{group}":
            entries.append((m, a, "away"))
    return entries


def verify_r32_labels(q) -> list[str]:
    """Confirm every R32 slot resolves and winners/runners match labels."""
    errors: list[str] = []
    seen: dict[int, int] = {}
    for m, h_label, a_label in R32_TEMPLATE:
        try:
            h_id = _resolve_label(h_label, q)
            a_id = _resolve_label(a_label, q)
        except Exception as exc:
            errors.append(f"M{m} resolve failed: {exc}")
            continue
        for tid in (h_id, a_id):
            seen[tid] = seen.get(tid, 0) + 1
        if h_label[0] == "1":
            g = h_label[1]
            if q.winners.get(g) != h_id:
                errors.append(f"M{m} home {h_label}: expected winner {q.winners.get(g)}, got {h_id}")
        if h_label[0] == "2":
            g = h_label[1]
            if q.runners_up.get(g) != h_id:
                errors.append(f"M{m} home {h_label}: expected runner-up {q.runners_up.get(g)}, got {h_id}")
        if a_label[0] == "1":
            g = a_label[1]
            if q.winners.get(g) != a_id:
                errors.append(f"M{m} away {a_label}: expected winner {q.winners.get(g)}, got {a_id}")
        if a_label[0] == "2":
            g = a_label[1]
            if q.runners_up.get(g) != a_id:
                errors.append(f"M{m} away {a_label}: expected runner-up {q.runners_up.get(g)}, got {a_id}")
        if h_label.startswith("3RD:"):
            slot = h_label.split(":", 1)[1]
            g = q.third_slots[slot][1]
            if q.third_by_group.get(g) != h_id:
                errors.append(f"M{m} third slot {h_label} mismatch")
        if a_label.startswith("3RD:"):
            slot = a_label.split(":", 1)[1]
            g = q.third_slots[slot][1]
            if q.third_by_group.get(g) != a_id:
                errors.append(f"M{m} third slot {a_label} mismatch")
    for tid, count in seen.items():
        if count != 1:
            errors.append(f"team_id={tid} appears in {count} R32 fixtures (expected 1)")
    return errors


def team_group_finish(ranked_by_group: dict[str, list[int]], tid: int) -> str:
    for g, ranked in ranked_by_group.items():
        if tid in ranked:
            pos = ranked.index(tid) + 1
            if pos <= 2:
                return f"{pos}{g}"
            if pos == 3:
                return f"3{g}"
            return "out"
    return "?"


def r32_match_for_team(q, tid: int) -> tuple[int, str, str] | None:
    """Return (match_no, home_label, away_label) for this team's R32 tie."""
    for m, h, a in R32_TEMPLATE:
        if _resolve_label(h, q) == tid or _resolve_label(a, q) == tid:
            return m, h, a
    return None


def match_participants(
    winners: dict[str, int], q, match_no: int
) -> tuple[int, int] | None:
    for template in (R32_TEMPLATE, R16_TEMPLATE, QF_TEMPLATE, SF_TEMPLATE, FINAL_TEMPLATE):
        for m, h, a in template:
            if m != match_no:
                continue
            if h.startswith("W"):
                home = winners.get(h)
            else:
                home = _resolve_label(h, q)
            if a.startswith("W"):
                away = winners.get(a)
            else:
                away = _resolve_label(a, q)
            if home is None or away is None:
                return None
            return int(home), int(away)
    return None


@dataclass
class TeamPathStats:
    n_sims: int = 0
    n_r32: int = 0
    n_sf: int = 0
    n_final: int = 0
    n_win: int = 0
    quarter: Counter = field(default_factory=Counter)
    group_finish: Counter = field(default_factory=Counter)
    r32_match: Counter = field(default_factory=Counter)
    sf_opponent: Counter = field(default_factory=Counter)
    final_opponent: Counter = field(default_factory=Counter)
    seeding_errors: int = 0


def trace_team(
    tid: int,
    q,
    ranked_by_group: dict[str, list[int]],
    winners: dict[str, int],
    champion: int,
) -> None:
    stats = path_stats[tid]
    stats.n_sims += 1
    finish = team_group_finish(ranked_by_group, tid)
    stats.group_finish[finish] += 1

    label_errors = verify_r32_labels(q)
    if label_errors:
        stats.seeding_errors += len(label_errors)

    r32 = r32_match_for_team(q, tid)
    if r32 is None:
        return
    stats.n_r32 += 1
    m_no = r32[0]
    stats.r32_match[m_no] += 1
    qf = R32_TO_QF.get(m_no, "?")
    stats.quarter[qf] += 1

    # SF: won 101 or 102
    sf_match = None
    for m in (101, 102):
        if winners.get(f"W{m}") == tid:
            sf_match = m
            break
    if sf_match is None:
        return
    stats.n_sf += 1
    parts = match_participants(winners, q, sf_match)
    if parts:
        h, a = parts
        opp = a if h == tid else h
        stats.sf_opponent[opp] += 1

    w101 = winners.get("W101")
    w102 = winners.get("W102")
    if tid not in (w101, w102):
        return
    stats.n_final += 1
    parts_f = match_participants(winners, q, 104)
    if parts_f:
        h, a = parts_f
        opp = a if h == tid else h
        stats.final_opponent[opp] += 1
    if champion == tid:
        stats.n_win += 1


def fit_analysis_model():
    """Same fitting path as ``run_monte_carlo`` (DB if populated, else history CSV)."""
    try:
        matches = load_training_matches()
        if matches:
            return fit_dixon_coles(matches, elo_by_team=load_latest_elo())
    except Exception:
        pass
    frame = load_history_frame(config.HISTORY_CSV_PATH)
    rows = [
        r
        for r in frame_to_rows(frame)
        if r["home_goals"] is not None and r["away_goals"] is not None
    ]
    elo, _ = fit_elo(rows)
    return fit_dixon_coles(rows, elo_by_team=elo)


def run_tracked_monte_carlo(n_sims: int, seed: int) -> None:
    global path_stats
    path_stats = {tid: TeamPathStats() for tid in FOCUS}

    teams = load_wc_teams()
    fixtures = load_group_fixtures()
    by_group = teams_by_group(teams)
    team_ids_by_group = {g: [t.team_id for t in ts] for g, ts in by_group.items()}
    wc_to_model = wc_id_to_model_id(teams)
    wc_team_ids = [t.team_id for t in teams]
    model = fit_analysis_model()
    catalog, layout = prepare_group_batch(
        model, fixtures, team_ids_by_group, wc_team_ids, wc_to_model
    )
    rng = np.random.default_rng(seed)
    seeding_global: list[str] = []

    home_goals, away_goals = sample_all_group_scorelines(
        catalog, fixtures, n_sims, rng
    )
    points, gf, ga = accumulate_standings(layout, home_goals, away_goals)
    idx_map = idx_to_team_id_map(layout)

    for s in range(n_sims):
        outcome = resolve_one_group_outcome(
            s,
            layout,
            team_ids_by_group,
            idx_map,
            points,
            gf,
            ga,
            home_goals,
            away_goals,
            rng,
        )
        q = build_qualifiers(
            outcome.ranked_by_group,
            outcome.third_by_group,
            outcome.best_eight_third_group_letters,
        )
        seeding_global.extend(verify_r32_labels(q))
        champion, winners = simulate_knockout(
            q, model, wc_to_model, rng, catalog=catalog
        )
        for tid in FOCUS:
            trace_team(
                tid,
                q,
                outcome.ranked_by_group,
                winners,
                champion,
            )

    for st in path_stats.values():
        st.n_sims = n_sims
    print(f"Tracked {n_sims} sims (seed={seed})\n")
    if seeding_global:
        print(f"SEEDING ERRORS: {len(seeding_global)} (first 5)")
        for e in seeding_global[:5]:
            print(f"  {e}")
    else:
        print("R32 label resolution: OK (every sim: 48 teams, one R32 slot each, 1X/2X/3RD match qualifiers)\n")

    print("Official R32 entry slots (static template) for focus teams:")
    for tid in (FRANCE, ENGLAND, PORTUGAL):
        print(f"  {NAMES[tid]}: {r32_entries_for_team(tid)}")
    print()

    for tid in (FRANCE, ENGLAND, PORTUGAL):
        st = path_stats[tid]
        print(f"=== {NAMES[tid]} (team_id={tid}) ===")
        p_final = st.n_final / n_sims
        p_win = st.n_win / n_sims
        p_win_given_final = st.n_win / st.n_final if st.n_final else float("nan")
        print(f"  P(reach final)     = {p_final:.4f}  ({st.n_final}/{n_sims})")
        print(f"  P(win title)     = {p_win:.4f}  ({st.n_win}/{n_sims})")
        print(f"  P(win | final)   = {p_win_given_final:.4f}  ({st.n_win}/{st.n_final})")
        print(f"  Most likely QF bracket: {st.quarter.most_common(1)[0] if st.quarter else 'n/a'}")
        print("  QF bracket distribution (among all sims):")
        for qf, c in st.quarter.most_common():
            print(f"    {qf}: {c/n_sims:.3f}")
        print("  Group finish distribution:")
        for lab, c in st.group_finish.most_common():
            print(f"    {lab}: {c/n_sims:.3f}")
        print("  R32 match_no distribution:")
        for m, c in st.r32_match.most_common(5):
            print(f"    M{m} -> {R32_TO_QF.get(m,'?')}: {c/n_sims:.3f}")
        print("  SF opponent (conditional on reaching SF):")
        for opp, c in st.sf_opponent.most_common(8):
            print(f"    vs {NAMES.get(opp, opp)}: {c/st.n_sf:.3f}")
        print("  Final opponent (conditional on reaching final):")
        for opp, c in st.final_opponent.most_common(10):
            print(f"    vs {NAMES.get(opp, opp)}: {c/st.n_final:.3f}")
        print()


def compare_mechanism() -> None:
    """Cross-check: France should face ARG/ESP more in finals with lower conditional win rate."""
    fr, eng, por = path_stats[FRANCE], path_stats[ENGLAND], path_stats[PORTUGAL]
    print("=== Mechanism check (conditional on reaching final) ===")
    for label, st in [("France", fr), ("England", eng), ("Portugal", por)]:
        if st.n_final == 0:
            continue
        p_arg = st.final_opponent[ARGENTINA] / st.n_final
        p_esp = st.final_opponent[SPAIN] / st.n_final
        p_win_gf = st.n_win / st.n_final
        print(
            f"  {label}: P(final vs Argentina)={p_arg:.3f} P(final vs Spain)={p_esp:.3f} "
            f"P(win|final)={p_win_gf:.3f}"
        )
    fr_p = fr.n_win / fr.n_final if fr.n_final else 0
    eng_p = eng.n_win / eng.n_final if eng.n_final else 0
    por_p = por.n_win / por.n_final if por.n_final else 0
    fr_boss = (fr.final_opponent[ARGENTINA] + fr.final_opponent[SPAIN]) / fr.n_final
    eng_boss = (eng.final_opponent[ARGENTINA] + eng.final_opponent[SPAIN]) / eng.n_final
    por_boss = (por.final_opponent[ARGENTINA] + por.final_opponent[SPAIN]) / por.n_final
    print()
    print(
        f"  France  P(win|final)={fr_p:.3f}  P(final vs ARG or ESP)={fr_boss:.3f}  "
        f"P(final)={fr.n_final / fr.n_sims:.3f}"
    )
    print(
        f"  England P(win|final)={eng_p:.3f}  P(final vs ARG or ESP)={eng_boss:.3f}  "
        f"P(final)={eng.n_final / eng.n_sims:.3f}"
    )
    print(
        f"  Portugal P(win|final)={por_p:.3f}  P(final vs ARG or ESP)={por_boss:.3f}  "
        f"P(final)={por.n_final / por.n_sims:.3f}"
    )
    mechanism_ok = (
        fr.n_final > eng.n_final
        and fr.n_final > por.n_final
        and fr_p < eng_p
        and fr_p < por_p
        and fr_boss > eng_boss
        and fr_boss > por_boss
    )
    print()
    if mechanism_ok:
        print(
            "VERDICT: Conditional-win pattern supports bracket-path explanation — "
            "France reaches the final more often but faces Argentina/Spain more "
            "frequently there and converts finals at a lower rate."
        )
    else:
        print(
            "VERDICT: Mechanism NOT confirmed by conditional opponents — "
            "investigate further (seeding error count above, or sampling)."
        )


path_stats: dict[int, TeamPathStats] = {}


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else config.N_SIMS
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else config.RANDOM_SEED
    global path_stats
    run_tracked_monte_carlo(n, seed)
    compare_mechanism()


if __name__ == "__main__":
    main()
