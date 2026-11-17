"""SQL quoting helpers. Every identifier and literal that reaches DuckDB goes through these."""
from __future__ import annotations


def q(name: str) -> str:
    """Double-quote an identifier, escaping embedded double quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def lit_str(value: str) -> str:
    """Single-quote a string literal, escaping embedded single quotes."""
    return "'" + str(value).replace("'", "''") + "'"
