"""Source- and category-balanced clip sampler.

Naive uniform sampling over all clips has an obvious failure mode when one
source (100STYLE, 800 clips) dominates another (CMU, 10 clips).  We enforce:

  1. PER-SOURCE target shares (independent of clip count)
  2. PER-CATEGORY target shares (locomotion/dance/fitness/etc.)

Two-stage rejection sampling: pick a category per target distribution, then
pick a clip from that category weighted by source shares.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .motion_loader import MotionClip
from .activity_tags import tag_clip, CATEGORIES


# Default weights — tuned so dance (AIST++ fills this) and locomotion get
# substantial shares, and "unknown" is small.
DEFAULT_SOURCE_WEIGHTS = {
    "cmu":      0.40,
    "100style": 0.25,
    "aistpp":   0.25,
    "mhad":     0.10,
}

DEFAULT_CATEGORY_WEIGHTS = {
    "locomotion": 0.22,
    "dance":      0.20,
    "fitness":    0.12,
    "martial":    0.08,
    "sports":     0.08,
    "gesture":    0.08,
    "daily":      0.10,
    "acrobatic":  0.08,
    "idle":       0.02,
    "unknown":    0.02,
}


@dataclass
class TaggedClip:
    clip: MotionClip
    category: str

    @property
    def source(self) -> str:
        return self.clip.source


@dataclass
class SamplerConfig:
    source_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SOURCE_WEIGHTS))
    category_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CATEGORY_WEIGHTS))
    fallback_any_category: bool = True


def _weighted_choice(items: list, weights: list[float], rng: random.Random):
    total = sum(weights)
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return items[-1]


def build_catalog(
    clips: list[MotionClip],
    cmu_descriptions: dict[str, str] | None = None,
) -> list[TaggedClip]:
    """Wrap each clip with its category tag."""
    return [
        TaggedClip(c, tag_clip(c.source, c.path, cmu_descriptions))
        for c in clips
    ]


def sample_plans(
    catalog: list[TaggedClip],
    n: int,
    rng: random.Random,
    cfg: SamplerConfig | None = None,
) -> list[TaggedClip]:
    """Return a list of `n` clip picks with source+category stratification."""
    cfg = cfg or SamplerConfig()
    by_category: dict[str, list[TaggedClip]] = {c: [] for c in CATEGORIES}
    for tc in catalog:
        by_category.setdefault(tc.category, []).append(tc)

    cats = [c for c in CATEGORIES if by_category.get(c)]
    cat_weights = [cfg.category_weights.get(c, 0.0) for c in cats]

    picks: list[TaggedClip] = []
    for _ in range(n):
        cat = _weighted_choice(cats, cat_weights, rng)
        pool = by_category[cat]
        if not pool and cfg.fallback_any_category:
            pool = catalog
        src_weights = [cfg.source_weights.get(tc.source, 0.01) for tc in pool]
        tc = _weighted_choice(pool, src_weights, rng)
        picks.append(tc)
    return picks


def report_distribution(picks: list[TaggedClip]) -> dict:
    """Empirical distribution of picks, for diversity reporting."""
    by_src: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    for tc in picks:
        by_src[tc.source] = by_src.get(tc.source, 0) + 1
        by_cat[tc.category] = by_cat.get(tc.category, 0) + 1
    n = max(1, len(picks))
    return {
        "n_picks": len(picks),
        "source_shares": {k: v / n for k, v in by_src.items()},
        "category_shares": {k: v / n for k, v in by_cat.items()},
    }
