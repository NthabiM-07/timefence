"""Load and validate the YAML feature registry."""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

AGGS = {"count", "sum", "max", "min", "avg"}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STR_LITERAL = re.compile(r"'(?:''|[^'])*'")


class RegistryError(ValueError):
    """The registry is malformed."""


class PermittedUseError(PermissionError):
    """A feature was requested for a purpose it is not approved for."""


@dataclass(frozen=True)
class Source:
    name: str
    path: str
    event_time: str
    ingested_at: str
    record_key: tuple[str, ...]
    version_col: str


@dataclass(frozen=True)
class Feature:
    name: str
    description: str
    owner: str
    permitted_use: tuple[str, ...]
    source: str | None = None
    agg: str | None = None
    column: str | None = None
    where: str | None = None
    window_days: int | None = None
    default: float | int | None = None
    derived: str | None = None
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_derived(self) -> bool:
        return self.derived is not None


@dataclass(frozen=True)
class Registry:
    entity: str
    sources: dict[str, Source]
    features: dict[str, Feature]
    sha256: str

    def select(self, names: list[str] | None = None, purpose: str | None = None) -> list[Feature]:
        """Features to build, in registry order, filtered by permitted use.

        Asking for a feature by name that is not approved for ``purpose`` is an
        error rather than a silent drop, so a model cannot quietly lose (or gain)
        an input.
        """
        if names is None:
            chosen = [f for f in self.features.values() if purpose is None or purpose in f.permitted_use]
        else:
            unknown = [n for n in names if n not in self.features]
            if unknown:
                raise RegistryError(f"Unknown features: {unknown}")
            chosen = [self.features[n] for n in names]
            if purpose is not None:
                blocked = [f.name for f in chosen if purpose not in f.permitted_use]
                if blocked:
                    raise PermittedUseError(f"Not approved for '{purpose}': {blocked}")
        for f in chosen:
            for dep in f.depends_on:
                d = self.features[dep]
                if purpose is not None and purpose not in d.permitted_use:
                    raise PermittedUseError(
                        f"Derived feature '{f.name}' depends on '{dep}', "
                        f"which is not approved for '{purpose}'"
                    )
        return chosen

    def closure(self, features: list[Feature]) -> list[Feature]:
        """The requested features plus everything they depend on, in registry order."""
        need: set[str] = set()
        stack = [f.name for f in features]
        while stack:
            n = stack.pop()
            if n not in need:
                need.add(n)
                stack.extend(self.features[n].depends_on)
        return [f for f in self.features.values() if f.name in need]


TOP_LEVEL_KEYS = {"entity", "sources", "features", "defaults"}  # `defaults` holds YAML anchors
SOURCE_KEYS = {"path", "event_time", "ingested_at", "record_key", "version_col"}
FEATURE_KEYS = {"name", "description", "owner", "permitted_use", "source", "agg", "column", "where",
                "window_days", "default", "derived"}
DERIVED_ONLY_KEYS = {"name", "description", "owner", "permitted_use", "derived"}
RESERVED_NAMES = {"decision_time"}  # plus the entity column; both are spine columns


def _reject_unknown_keys(where: str, given: dict, allowed: set[str]) -> None:
    """Fail loudly on misspelt keys: a typo like `windows_days` must not silently drop a window."""
    for key in given:
        if key not in allowed:
            close = difflib.get_close_matches(str(key), sorted(allowed), n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            raise RegistryError(f"{where}: unknown key '{key}'{hint}")


def _str_list(where: str, value, example: str) -> tuple[str, ...]:
    """A non-empty list of strings. A bare string is an error, not a list of its characters."""
    if (not isinstance(value, list) or not value
            or not all(isinstance(v, str) and v for v in value)):
        raise RegistryError(f"{where} must be a non-empty list of names, e.g. [{example}]")
    return tuple(value)


def _sql_fragment(where: str, value) -> str:
    text = str(value)
    if ";" in text:
        raise RegistryError(f"{where} may not contain ';' (one expression, not a statement)")
    return text


def load_registry(path: str | Path) -> Registry:
    text = Path(path).read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or "features" not in raw or "sources" not in raw:
        raise RegistryError("Registry needs 'entity', 'sources' and 'features'")
    _reject_unknown_keys("Registry", raw, TOP_LEVEL_KEYS)
    entity = raw.get("entity")
    if not entity or not isinstance(entity, str):
        raise RegistryError("Registry needs an 'entity' column name")
    if not isinstance(raw["sources"], dict) or not isinstance(raw["features"], list):
        raise RegistryError("'sources' must be a mapping and 'features' a list")

    sources: dict[str, Source] = {}
    for name, s in raw["sources"].items():
        where = f"Source '{name}'"
        if not isinstance(s, dict):
            raise RegistryError(f"{where} must be a mapping")
        _reject_unknown_keys(where, s, SOURCE_KEYS)
        missing = SOURCE_KEYS - set(s)
        if missing:
            raise RegistryError(f"{where} is missing {sorted(missing)}")
        sources[name] = Source(name=name, path=s["path"], event_time=s["event_time"],
                               ingested_at=s["ingested_at"],
                               record_key=_str_list(f"{where} record_key", s["record_key"], "account_id"),
                               version_col=s["version_col"])

    features: dict[str, Feature] = {}
    names = [f.get("name") if isinstance(f, dict) else None for f in raw["features"]]
    dupes = sorted({n for n in names if n is not None and names.count(n) > 1})
    if dupes:
        raise RegistryError(f"Duplicate feature names: {dupes}")
    for f in raw["features"]:
        if not isinstance(f, dict):
            raise RegistryError(f"Every feature must be a mapping, got {f!r}")
        name = f.get("name")
        if not isinstance(name, str) or not _IDENT.fullmatch(name) or name.startswith("__"):
            raise RegistryError(f"Bad feature name: {name!r}")
        if name in RESERVED_NAMES | {entity}:
            raise RegistryError(f"Feature name '{name}' is reserved for a spine column")
        where = f"Feature '{name}'"
        _reject_unknown_keys(where, f, DERIVED_ONLY_KEYS if "derived" in f else FEATURE_KEYS)
        for req in ("description", "owner", "permitted_use"):
            if not f.get(req):
                raise RegistryError(f"{where} needs '{req}' (governance metadata is mandatory)")
        common = dict(name=name, description=f["description"], owner=f["owner"],
                      permitted_use=_str_list(f"{where} permitted_use", f["permitted_use"],
                                              "credit_decisioning, analytics"))
        if "derived" in f:
            expr = _sql_fragment(f"{where} derived", f["derived"])
            # Strip string literals first: `CASE WHEN x = 'enquiries_3m'` is not a dependency.
            tokens = _IDENT.findall(_STR_LITERAL.sub(" ", expr))
            deps = tuple(t for t in dict.fromkeys(tokens) if t in names)
            if not deps:
                raise RegistryError(f"Derived feature '{name}' references no other feature")
            features[name] = Feature(**common, derived=expr, depends_on=deps)
            continue
        if f.get("source") not in sources:
            raise RegistryError(f"{where} has unknown source {f.get('source')!r}")
        if f.get("agg") not in AGGS:
            raise RegistryError(f"{where} has agg {f.get('agg')!r}; allowed {sorted(AGGS)}")
        if "column" not in f:
            raise RegistryError(f"{where} needs a 'column' expression")
        w = f.get("window_days")
        if w is not None and (isinstance(w, bool) or not isinstance(w, int) or w <= 0):
            raise RegistryError(f"{where} window_days must be a positive integer")
        d = f.get("default")
        if d is not None and (isinstance(d, bool) or not isinstance(d, (int, float))):
            raise RegistryError(f"{where} default must be a number")
        features[name] = Feature(
            **common, source=f["source"], agg=f["agg"],
            column=_sql_fragment(f"{where} column", f["column"]),
            where=_sql_fragment(f"{where} where", f["where"]) if f.get("where") else None,
            window_days=w, default=d)

    _check_acyclic(features)
    canonical = json.dumps(
        {"entity": entity,
         "sources": {k: asdict(v) for k, v in sources.items()},
         "features": {k: asdict(v) for k, v in features.items()}},
        sort_keys=True, default=list,
    )
    return Registry(entity=entity, sources=sources, features=features,
                    sha256=hashlib.sha256(canonical.encode()).hexdigest())


def _check_acyclic(features: dict[str, Feature]) -> None:
    state: dict[str, int] = {}

    def visit(n: str, path: tuple[str, ...]) -> None:
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise RegistryError(f"Cycle in derived features: {' -> '.join((*path, n))}")
        state[n] = 1
        for d in features[n].depends_on:
            visit(d, (*path, n))
        state[n] = 2

    for n in features:
        visit(n, ())


def derived_layers(features: list[Feature]) -> list[list[Feature]]:
    """Group derived features so each layer only depends on earlier ones."""
    done = {f.name for f in features if not f.is_derived}
    todo = [f for f in features if f.is_derived]
    layers = []
    while todo:
        layer = [f for f in todo if all(d in done for d in f.depends_on)]
        if not layer:
            raise RegistryError("Unresolvable derived features: " + ", ".join(f.name for f in todo))
        layers.append(layer)
        done |= {f.name for f in layer}
        todo = [f for f in todo if f not in layer]
    return layers
