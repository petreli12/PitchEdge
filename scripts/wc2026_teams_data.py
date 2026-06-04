"""Confirmed WC 2026 group-stage teams (post-draw, March 2026).

Sources: FIFA draw / widely published group lists (NBC, Roadtrips, Olympics.com).
Verify against https://www.fifa.com if anything changed.

Each entry: (name, fifa_code, confederation, odds_name)
``odds_name`` is the string to match the-odds-api ``home_team`` / ``away_team`` when
it differs from ``name``. Leave empty to use ``name``.

``HISTORY_NAME_OVERRIDES`` maps display ``name`` → Kaggle ``results.csv`` label when
they differ (used for Dixon-Coles ``model_team_id`` and ``score.py`` lookups).
"""

from __future__ import annotations

# Display name -> Kaggle home_team/away_team string (only where != ``name``).
HISTORY_NAME_OVERRIDES: dict[str, str] = {
    "Czechia": "Czech Republic",
    "Turkiye": "Turkey",
    "Curacao": "Curaçao",
}

# (name, fifa_code, confederation, odds_name or "")
GROUPS: dict[str, list[tuple[str, str, str, str]]] = {
    "A": [
        ("Mexico", "MEX", "CONCACAF", ""),
        ("South Africa", "RSA", "CAF", ""),
        ("South Korea", "KOR", "AFC", ""),
        ("Czechia", "CZE", "UEFA", "Czech Republic"),
    ],
    "B": [
        ("Canada", "CAN", "CONCACAF", ""),
        ("Switzerland", "SUI", "UEFA", ""),
        ("Qatar", "QAT", "AFC", ""),
        ("Bosnia and Herzegovina", "BIH", "UEFA", "Bosnia & Herzegovina"),
    ],
    "C": [
        ("Brazil", "BRA", "CONMEBOL", ""),
        ("Morocco", "MAR", "CAF", ""),
        ("Scotland", "SCO", "UEFA", ""),
        ("Haiti", "HAI", "CONCACAF", ""),
    ],
    "D": [
        ("United States", "USA", "CONCACAF", "USA"),
        ("Paraguay", "PAR", "CONMEBOL", ""),
        ("Australia", "AUS", "AFC", ""),
        ("Turkiye", "TUR", "UEFA", "Turkey"),
    ],
    "E": [
        ("Germany", "GER", "UEFA", ""),
        ("Ecuador", "ECU", "CONMEBOL", ""),
        ("Ivory Coast", "CIV", "CAF", ""),
        ("Curacao", "CUW", "CONCACAF", "Curaçao"),
    ],
    "F": [
        ("Netherlands", "NED", "UEFA", ""),
        ("Japan", "JPN", "AFC", ""),
        ("Tunisia", "TUN", "CAF", ""),
        ("Sweden", "SWE", "UEFA", ""),
    ],
    "G": [
        ("Belgium", "BEL", "UEFA", ""),
        ("Iran", "IRN", "AFC", ""),
        ("Egypt", "EGY", "AFC", ""),
        ("New Zealand", "NZL", "OFC", ""),
    ],
    "H": [
        ("Spain", "ESP", "UEFA", ""),
        ("Uruguay", "URU", "CONMEBOL", ""),
        ("Saudi Arabia", "KSA", "AFC", ""),
        ("Cape Verde", "CPV", "CAF", ""),
    ],
    "I": [
        ("France", "FRA", "UEFA", ""),
        ("Senegal", "SEN", "CAF", ""),
        ("Norway", "NOR", "UEFA", ""),
        ("Iraq", "IRQ", "AFC", ""),
    ],
    "J": [
        ("Argentina", "ARG", "CONMEBOL", ""),
        ("Austria", "AUT", "UEFA", ""),
        ("Algeria", "ALG", "CAF", ""),
        ("Jordan", "JOR", "AFC", ""),
    ],
    "K": [
        ("Portugal", "POR", "UEFA", ""),
        ("Colombia", "COL", "CONMEBOL", ""),
        ("Uzbekistan", "UZB", "AFC", ""),
        ("DR Congo", "COD", "CAF", ""),
    ],
    "L": [
        ("England", "ENG", "UEFA", ""),
        ("Croatia", "CRO", "UEFA", ""),
        ("Panama", "PAN", "CONCACAF", ""),
        ("Ghana", "GHA", "CAF", ""),
    ],
}
