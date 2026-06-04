"""Venue / home-advantage policy for Dixon-Coles prediction calls.

PitchEdge product scope (WC 2026 + historical international tournaments):
almost every fixture is played on a neutral pitch in the modeling sense — ``γ``
is off (``neutral=True`` on ``match_probs`` / ``score_matrix``).

**Host-nation exception (WC 2026, intentional):** When a co-host is listed as the
``home_id`` in ``fixtures`` (USA ``13``, Mexico ``1``, Canada ``5``), we set
``neutral=False`` so ``γ`` applies to that home slot. This is a deliberate
stand-in until fixtures carry an explicit ``host_home`` column tied to city/
stadium. Knockout and group matches between two non-hosts remain neutral even if
played in the United States.

Phase 4 backtests on past World Cups / Euros / Copa América must use
``neutral=True`` for every match (no co-host exception in those replays unless
you pass ``host_home=True`` explicitly).
"""

from __future__ import annotations

# ``teams.team_id`` for WC 2026 co-hosts (see data/wc2026/teams.csv).
WC_2026_HOST_TEAM_IDS: frozenset[int] = frozenset({1, 5, 13})  # Mexico, Canada, USA


def is_wc_2026_host(team_id: int) -> bool:
    """True if ``team_id`` is USA, Mexico, or Canada in the WC 2026 squad table."""
    return team_id in WC_2026_HOST_TEAM_IDS


def dixon_coles_neutral_for_wc_fixture(
    home_team_id: int,
    *,
    host_home: bool | None = None,
) -> bool:
    """Return the ``neutral`` flag for Dixon-Coles APIs on a WC-style fixture.

    Parameters
    ----------
    home_team_id:
        ``fixtures.home_id`` (WC ``teams.team_id``, not historical hash ids).
    host_home:
        When ``True``, force ``neutral=False`` (apply ``γ``). When ``False``,
        force ``neutral=True``. When ``None`` (default), apply ``γ`` only if
        ``home_team_id`` is a 2026 co-host — see module docstring.
    """
    if host_home is True:
        return False
    if host_home is False:
        return True
    return home_team_id not in WC_2026_HOST_TEAM_IDS


def dixon_coles_neutral_for_tournament_fixture(
    *,
    host_home: bool = False,
) -> bool:
    """``neutral`` flag for historical international-tournament backtests.

    All held-out cup matches are neutral venues unless ``host_home=True`` is set
    deliberately (e.g. a rare host-nation home match in the training era).
    """
    return not host_home
