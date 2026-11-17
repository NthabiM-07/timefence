"""Load and validate the YAML feature registry."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

AGGS = {"count", "sum", "max", "min", "avg"}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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
                        f"Derived feature '{f.name}' depends on '{dep}', which is not approved for '{purpose}'"
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


def load_registry(path: str | Path) -> Registry:
    text = Path(path).read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or "features" not in raw or "sources" not in raw:
        raise RegistryError("Registry needs 'entity', 'sources' and 'features'")
    entity = raw.get("entity")
    if not entity:
        raise RegistryError("Registry needs an 'entity'")

    sources: dict[str, Source] = {}
    for name, s in raw["sources"].items():
        missing = {"path", "event_time", "ingested_at", "record_key", "version_col"} - set(s)
        if missing:
            raise RegistryError(f"Source '{name}' is missing {sorted(missing)}")
        sources[name] = Source(name=name, path=s["path"], event_time=s["event_time"],
                               ingested_at=s["ingested_at"], record_key=tuple(s["record_key"]),
                               version_col=s["version_col"])

    features: dict[str, Feature] = {}
    names = [f.get("name") for f in raw["features"]]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise RegistryError(f"Duplicate feature names: {dupes}")
    for f in raw["features"]:
        name = f.get("name")
        if not name or not _IDENT.fullmatch(name) or name.startswith("__"):
            raise RegistryError(f"Bad feature name: {name!r}")
        for req in ("description", "owner", "permitted_use"):
            if not f.get(req):
                raise RegistryError(f"Feature '{name}' needs '{req}' (governance metadata is mandatory)")
        common = dict(name=name, description=f["description"], owner=f["owner"],
                      permitted_use=tuple(f["permitted_use"]))
        if "derived" in f:
            deps = tuple(t for t in dict.fromkeys(_IDENT.findall(f["derived"])) if t in names)
            if not deps:
                raise RegistryError(f"Derived feature '{name}' references no other feature")
            features[name] = Feature(**common, derived=f["derived"], depends_on=deps)
            continue
        if f.get("source") not in sources:
            raise RegistryError(f"Feature '{name}' has unknown source {f.get('source')!r}")
        if f.get("agg") not in AGGS:
            raise RegistryError(f"Feature '{name}' has agg {f.get('agg')!r}; allowed {sorted(AGGS)}")
        if "column" not in f:
            raise RegistryError(f"Feature '{name}' needs a 'column' expression")
        w = f.get("window_days")
        if w is not None and (not isinstance(w, int) or w <= 0):
            raise RegistryError(f"Feature '{name}' window_days must be a positive integer")
        features[name] = Feature(**common, source=f["source"], agg=f["agg"], column=str(f["column"]),
                                 where=f.get("where"), window_days=w, default=f.get("default"))

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
            raise RegistryError(f"Cycle in derived features: {' -> '.join(path + (n,))}")
        state[n] = 1
        for d in features[n].depends_on:
            visit(d, path + (n,))
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
