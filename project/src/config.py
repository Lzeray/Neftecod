"""Configuration loading (paths, horizons, weights, seeds).

Paths in ``config/config.yaml`` are relative to ``paths.data_root``, itself
relative to the project directory.  Nothing in the code base hard-codes a data
path (prompt §1).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any]
    project_dir: Path = field(default=PROJECT_DIR)

    # ---------------- dictionary-style access ----------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # ---------------- paths ----------------
    @property
    def data_root(self) -> Path:
        root = Path(self.raw["paths"]["data_root"])
        return root if root.is_absolute() else (self.project_dir / root).resolve()

    def data_path(self, key: str) -> Path:
        """Resolve a key of ``paths:`` against ``data_root``."""
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else (self.data_root / p).resolve()

    @property
    def artifacts_dir(self) -> Path:
        p = Path(self.raw["paths"]["artifacts"])
        p = p if p.is_absolute() else (self.project_dir / p).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_dir(self) -> Path:
        p = Path(self.raw["paths"]["cache"])
        p = p if p.is_absolute() else (self.project_dir / p).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def Ts(self) -> float:
        """Sampling period, seconds."""
        return float(self.raw["time"]["Ts_s"])


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path is not None else PROJECT_DIR / "config" / "config.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw=raw)
    set_seed(int(raw.get("seed", 0)))
    return cfg


def set_seed(seed: int) -> None:
    """Determinism: every entry point calls this (prompt §2)."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
