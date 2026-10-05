"""Canonical club identities shared by the Sofascore- and WhoScored-backed modules.

The two data sources label the same club differently — WhoScored says "PSG",
"Wolves", "Rennes"; Sofascore says "Paris Saint-Germain", "Wolverhampton",
"Stade Rennais". A token-overlap heuristic cannot bridge an acronym to its
expansion and simultaneously keep "Real Madrid" apart from "Real Betis", so
every label either source emits is enumerated here instead, together with the
nicknames a user is likely to type.

`teams_match` therefore decides by canonical id whenever both sides are known
labels, and only falls back to fuzzy comparison for free text that isn't in the
table. That fallback deliberately omits the old "any shared token" rule, which
made "Man Utd" match Leeds United, Newcastle United and West Ham United.

Add new clubs as a new group whose first entry is the Sofascore label, since
that is what the UI displays.
"""

from __future__ import annotations

import re

from name_utils import normalize_name, strip_accents

# First entry of each group is canonical (the Sofascore label); the rest are
# WhoScored labels and common user spellings/nicknames.
ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    # England - Premier League
    ("Arsenal", "Arsenal FC", "Gunners"),
    ("Aston Villa", "Villa"),
    ("Bournemouth", "AFC Bournemouth"),
    ("Brentford", "Brentford FC"),
    ("Brighton & Hove Albion", "Brighton", "Brighton and Hove Albion"),
    ("Burnley", "Burnley FC"),
    ("Chelsea", "Chelsea FC"),
    ("Crystal Palace", "Palace"),
    ("Everton", "Everton FC"),
    ("Fulham", "Fulham FC"),
    ("Leeds United", "Leeds"),
    ("Liverpool FC", "Liverpool"),
    ("Manchester City", "Man City", "Man. City", "MCFC"),
    ("Manchester United", "Man Utd", "Man United", "MUFC"),
    ("Newcastle United", "Newcastle"),
    ("Nottingham Forest", "Nott'm Forest", "Notts Forest"),
    ("Sunderland", "Sunderland AFC"),
    ("Tottenham Hotspur", "Tottenham", "Spurs"),
    ("West Ham United", "West Ham"),
    ("Wolverhampton", "Wolves", "Wolverhampton Wanderers"),
    # Spain - La Liga
    ("Athletic Club", "Athletic Bilbao", "Athletic"),
    ("Atlético Madrid", "Atletico", "Atletico Madrid", "Atleti"),
    ("FC Barcelona", "Barcelona", "Barca"),
    ("Celta Vigo", "Celta", "RC Celta"),
    ("Deportivo Alavés", "Deportivo Alaves", "Alaves"),
    ("Elche", "Elche CF"),
    ("Espanyol", "RCD Espanyol"),
    ("Getafe", "Getafe CF"),
    ("Girona FC", "Girona"),
    ("Levante UD", "Levante"),
    ("Mallorca", "RCD Mallorca"),
    ("Osasuna", "CA Osasuna"),
    ("Rayo Vallecano", "Rayo"),
    ("Real Betis", "Betis"),
    ("Real Madrid", "Real Madrid CF"),
    ("Real Oviedo", "Oviedo"),
    ("Real Sociedad", "Sociedad"),
    ("Sevilla", "Sevilla FC"),
    ("Valencia", "Valencia CF"),
    ("Villarreal", "Villarreal CF"),
    # France - Ligue 1
    ("Angers", "Angers SCO"),
    ("Auxerre", "AJ Auxerre"),
    ("Stade Brestois", "Brest", "Stade Brestois 29"),
    ("Le Havre", "Le Havre AC"),
    ("RC Lens", "Lens"),
    ("Lille", "LOSC Lille", "LOSC"),
    ("Lorient", "FC Lorient"),
    ("Olympique Lyonnais", "Lyon", "OL"),
    ("Olympique de Marseille", "Marseille", "OM"),
    ("Metz", "FC Metz"),
    ("AS Monaco", "Monaco"),
    ("Nantes", "FC Nantes"),
    ("Nice", "OGC Nice"),
    # Paris FC is a separate club and must never collide with PSG.
    ("Paris Saint-Germain", "PSG", "Paris SG", "Paris Saint Germain"),
    ("Paris FC",),
    ("Stade Rennais", "Rennes", "Stade Rennais FC"),
    ("RC Strasbourg", "Strasbourg", "RC Strasbourg Alsace"),
    ("Toulouse", "Toulouse FC"),
    ("Saint-Étienne", "Saint-Etienne", "ASSE"),
    ("Rodez AF", "Rodez"),
    # Germany - Bundesliga
    ("FC Augsburg", "Augsburg"),
    ("FC Bayern München", "Bayern", "Bayern Munich", "Bayern München", "FC Bayern"),
    ("Borussia Dortmund", "Dortmund", "BVB"),
    (
        "Borussia M'gladbach",
        "Borussia M.Gladbach",
        "Borussia Mönchengladbach",
        "Borussia Monchengladbach",
        "Gladbach",
    ),
    ("Eintracht Frankfurt", "Frankfurt"),
    ("1. FC Heidenheim", "FC Heidenheim", "Heidenheim"),
    ("1. FC Köln", "FC Koln", "FC Köln", "Köln", "Cologne"),
    ("1. FC Union Berlin", "Union Berlin"),
    ("1. FSV Mainz 05", "Mainz", "Mainz 05"),
    ("Bayer 04 Leverkusen", "Leverkusen", "Bayer Leverkusen"),
    ("FC St. Pauli", "St. Pauli", "St Pauli"),
    ("Hamburger SV", "Hamburg", "HSV"),
    ("RB Leipzig", "RBL", "Leipzig"),
    ("SC Freiburg", "Freiburg"),
    ("SV Werder Bremen", "Werder Bremen", "Werder", "Bremen"),
    ("TSG Hoffenheim", "Hoffenheim"),
    ("VfB Stuttgart", "Stuttgart"),
    ("VfL Wolfsburg", "Wolfsburg"),
    # Italy - Serie A
    ("AC Milan", "Milan"),
    ("AS Roma", "Roma"),
    ("Atalanta", "Atalanta BC"),
    ("Bologna", "Bologna FC"),
    ("Cagliari", "Cagliari Calcio"),
    ("Como", "Como 1907"),
    ("Cremonese", "US Cremonese"),
    ("Fiorentina", "ACF Fiorentina"),
    ("Genoa", "Genoa CFC"),
    ("Hellas Verona", "Verona"),
    ("Inter", "Inter Milan", "Internazionale"),
    ("Juventus", "Juve"),
    ("Lazio", "SS Lazio"),
    ("Lecce", "US Lecce"),
    ("Parma", "Parma Calcio 1913", "Parma Calcio"),
    ("Pisa", "Pisa SC"),
    ("SSC Napoli", "Napoli"),
    ("Sassuolo", "US Sassuolo"),
    ("Torino", "Torino FC"),
    ("Udinese", "Udinese Calcio"),
    # Netherlands - Eredivisie
    ("AFC Ajax", "Ajax"),
    ("AZ Alkmaar", "AZ"),
    ("Almere City FC", "Almere City", "Almere"),
    ("De Graafschap", "Graafschap"),
    ("Excelsior", "SBV Excelsior"),
    ("FC Groningen", "Groningen"),
    ("FC Twente", "Twente"),
    ("FC Utrecht", "Utrecht"),
    ("FC Volendam", "Volendam"),
    ("Feyenoord", "Feyenoord Rotterdam"),
    ("Fortuna Sittard", "Fortuna"),
    ("Go Ahead Eagles", "Go Ahead"),
    ("Heracles Almelo", "Heracles"),
    ("NAC Breda", "NAC"),
    ("NEC Nijmegen", "NEC", "Nijmegen"),
    ("PEC Zwolle", "PEC", "Zwolle"),
    ("PSV Eindhoven", "PSV"),
    ("SC Heerenveen", "Heerenveen"),
    ("SC Telstar", "Telstar"),
    ("Sparta Rotterdam", "Sparta"),
    ("Willem II Tilburg", "Willem II"),
    # Portugal - Primeira Liga
    ("AVS - Futebol SAD", "AVS Futebol SAD", "AVS"),
    ("Benfica", "SL Benfica"),
    ("CD Nacional", "Nacional"),
    ("CF Estrela Amadora", "Estrela da Amadora", "Estrela Amadora"),
    ("Casa Pia", "Casa Pia AC"),
    ("Estoril Praia", "Estoril"),
    ("FC Alverca", "Alverca"),
    ("FC Arouca", "Arouca"),
    ("FC Porto", "Porto"),
    ("Famalicão", "Famalicao", "FC Famalicao"),
    ("Gil Vicente", "Gil Vicente FC"),
    ("Moreirense", "Moreirense FC"),
    ("Rio Ave", "Rio Ave FC"),
    ("Santa Clara", "CD Santa Clara"),
    ("Sporting Braga", "Braga", "SC Braga"),
    ("Sporting CP", "Sporting", "Sporting Lisbon"),
    ("Tondela", "CD Tondela"),
    ("Vitória SC", "Vitoria de Guimaraes", "Vitória de Guimarães", "Vitoria Guimaraes"),
)


def squash(name: str | None) -> str:
    """Accent-fold, lower-case and strip every non-alphanumeric character.

    "Utd" is rewritten to "United" first so the WhoScored and Sofascore
    spellings of the Manchester/Leeds/Newcastle clubs collapse together.
    """
    if not name:
        return ""
    text = strip_accents(str(name)).lower()
    text = re.sub(r"\butd\b", "united", text)
    return re.sub(r"[^a-z0-9]", "", text)


def _build_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for group in ALIAS_GROUPS:
        canonical = normalize_name(group[0])
        for label in group:
            key = squash(label)
            if not key:
                continue
            existing = lookup.get(key)
            if existing is not None and existing != canonical:
                raise ValueError(
                    f"team alias {label!r} maps to both {existing!r} and {canonical!r}"
                )
            lookup[key] = canonical
    return lookup


_LOOKUP = _build_lookup()


def canonical_team(name: str | None) -> str | None:
    """Canonical id for a known club label, or None for unrecognized text."""
    return _LOOKUP.get(squash(name))


def display_team(name: str | None) -> str | None:
    """Preferred (Sofascore) spelling for a known label, else the input unchanged."""
    canonical = canonical_team(name)
    if canonical is None:
        return name
    for group in ALIAS_GROUPS:
        if normalize_name(group[0]) == canonical:
            return group[0]
    return name


def _tokens(name: str) -> frozenset[str]:
    text = strip_accents(str(name)).lower()
    text = re.sub(r"\butd\b", "united", text)
    return frozenset(re.findall(r"[a-z0-9]+", text))


def teams_match(a: str | None, b: str | None) -> bool:
    """True when two club labels plausibly refer to the same club.

    An unset label matches anything, since `team=` is an optional narrowing
    hint everywhere it is used rather than a required filter.
    """
    if not a or not b:
        return True

    canon_a, canon_b = canonical_team(a), canonical_team(b)
    if canon_a is not None and canon_b is not None:
        return canon_a == canon_b

    squashed_a, squashed_b = squash(a), squash(b)
    if not squashed_a or not squashed_b:
        return True
    if squashed_a == squashed_b:
        return True

    # A strict token subset covers abbreviations of an unknown club's full
    # name ("Heracles" vs "Heracles Almelo") without letting two clubs match
    # on one shared word ("Real Madrid" vs "Real Betis").
    tokens_a, tokens_b = _tokens(a), _tokens(b)
    if tokens_a and tokens_b and (tokens_a < tokens_b or tokens_b < tokens_a):
        return True

    return squashed_a in squashed_b or squashed_b in squashed_a
