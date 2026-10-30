"""Immutable strategy-version registry used by deployment plans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "config" / "strategy-registry.json"
FAMILIES = frozenset({"bargaining", "negotiation", "persuasion"})
ROLES = frozenset({"primary", "alternate"})
UNKNOWN = "unknown"


@dataclass(frozen=True)
class StrategyVersion:
    strategy_id: str
    family: str
    immutable_commit: str
    source_path: str
    deploy_path: str
    status: str
    role: str
    evidence: Any

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StrategyVersion":
        if not isinstance(raw, dict):
            raise ValueError("strategy registry entries must be objects")
        required = (
            "strategy_id",
            "family",
            "immutable_commit",
            "source_path",
            "deploy_path",
            "status",
            "role",
            "evidence",
        )
        for key in required:
            if key not in raw or raw[key] in (None, ""):
                raise ValueError(f"strategy registry entry missing {key}")
        family = str(raw["family"])
        if family not in FAMILIES:
            raise ValueError(f"invalid strategy family: {family}")
        role = str(raw["role"])
        if role not in ROLES:
            raise ValueError(f"invalid strategy role: {role}")
        return cls(
            strategy_id=str(raw["strategy_id"]),
            family=family,
            immutable_commit=str(raw["immutable_commit"]),
            source_path=str(raw["source_path"]),
            deploy_path=str(raw["deploy_path"]),
            status=str(raw["status"]),
            role=role,
            evidence=raw["evidence"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "family": self.family,
            "immutable_commit": self.immutable_commit,
            "source_path": self.source_path,
            "deploy_path": self.deploy_path,
            "status": self.status,
            "role": self.role,
            "evidence": self.evidence,
        }


class StrategyRegistry:
    """Validated registry with one-to-many family membership."""

    def __init__(self, versions: Iterable[StrategyVersion]):
        values = tuple(versions)
        by_id: dict[str, StrategyVersion] = {}
        primaries: dict[str, StrategyVersion] = {}
        for version in values:
            if version.strategy_id in by_id:
                raise ValueError(f"duplicate strategy_id: {version.strategy_id}")
            by_id[version.strategy_id] = version
            if version.role == "primary":
                if version.family in primaries:
                    raise ValueError(f"family {version.family} already has a primary")
                primaries[version.family] = version
        self._versions = values
        self._by_id = by_id
        self._primaries = primaries

    @property
    def versions(self) -> tuple[StrategyVersion, ...]:
        return self._versions

    def for_family(self, family: str) -> tuple[StrategyVersion, ...]:
        return tuple(version for version in self._versions if version.family == family)

    def primary(self, family: str) -> StrategyVersion:
        try:
            return self._primaries[family]
        except KeyError as exc:
            raise ValueError(f"family {family} has no primary") from exc

    def get(self, strategy_id: str) -> StrategyVersion:
        try:
            return self._by_id[strategy_id]
        except KeyError as exc:
            raise ValueError(f"unknown strategy_id: {strategy_id}") from exc

    def resolve(self, strategy_id: str, family: str, commit: str) -> StrategyVersion:
        version = self.get(strategy_id)
        if version.family != family:
            raise ValueError(
                f"strategy {strategy_id} belongs to {version.family}, not {family}"
            )
        if version.immutable_commit != commit:
            raise ValueError(
                f"strategy {strategy_id}: commit does not match immutable_commit"
            )
        return version


def load_registry(path: Path = DEFAULT_PATH) -> StrategyRegistry:
    """Load and validate a registry JSON file."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read strategy registry: {path}") from exc
    rows = data.get("strategies") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("strategy registry requires a non-empty strategies list")
    return StrategyRegistry(StrategyVersion.from_dict(row) for row in rows)

def default_registry() -> StrategyRegistry:
    return load_registry()
