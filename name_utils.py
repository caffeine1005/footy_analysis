"""Accent/diacritic-insensitive name matching.

Lets you type player, team and league names in the plain English alphabet and
still match the accented spelling stored in the data, e.g. "Arda Guler" ->
"Arda Güler", "Odegaard" -> "Ødegaard", "Atletico" -> "Atlético".

`normalize_name` folds a single Python string. `normalize_series` folds a
pandas Series, and `normalized_col` builds the equivalent Polars expression, so
pandas- and polars-backed lookups behave identically.
"""

from __future__ import annotations

import unicodedata

# Characters that don't decompose under NFKD but have an obvious ASCII fold.
_TRANSLIT = {
    "ø": "o", "Ø": "O",
    "ł": "l", "Ł": "L",
    "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "Th",
    "ß": "ss",
    "æ": "ae", "Æ": "Ae",
    "œ": "oe", "Œ": "Oe",
    "ı": "i", "İ": "I",
    "ŋ": "n", "Ŋ": "N",
}


def strip_accents(text: str) -> str:
    """Fold accented/diacritic characters to their plain ASCII base."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFKD", str(text)):
        if unicodedata.combining(ch):
            continue
        out.append(_TRANSLIT.get(ch, ch))
    return "".join(out)


def normalize_name(text: str | None) -> str:
    """Lower-cased, accent-folded, trimmed form used for comparisons."""
    if text is None:
        return ""
    return strip_accents(text).casefold().strip()


def normalize_series(series):
    """Accent-fold a pandas Series of names for elementwise comparison."""
    return series.fillna("").map(normalize_name)


def normalized_col(col):
    """Polars expression that accent-folds and lower-cases a string column.

    `col` may be a column name or an existing `pl.Expr`.
    """
    import polars as pl

    expr = pl.col(col) if isinstance(col, str) else col
    expr = expr.str.normalize("NFKD").str.replace_all(r"\p{M}", "")
    for src, dst in _TRANSLIT.items():
        expr = expr.str.replace_all(src, dst, literal=True)
    return expr.str.to_lowercase()
