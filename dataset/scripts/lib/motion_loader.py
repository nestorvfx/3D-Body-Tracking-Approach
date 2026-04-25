"""Unified motion source loader.  Enumerates BVH files across commercial-clean
sources (CMU, 100STYLE, AIST++ converted, Berkeley MHAD converted) with
per-source licensing metadata preserved.

Each source has its own skeleton conventions; the loader tags each clip so
the retargeter can dispatch to the correct bone-mapping table.

Supported sources:
  - cmu     : CMU Mocap cgspeed/daz BVH (public domain)
  - 100style: 100STYLE BVH (CC-BY 4.0)
  - aistpp  : AIST++ SMPL params converted to BVH (CC-BY 4.0 annotations)
  - mhad    : Berkeley MHAD converted to BVH (BSD-2)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


SOURCE_LICENSE = {
    "cmu":      "Public Domain (free for any use) — mocap.cs.cmu.edu",
    "100style": "CC-BY 4.0 — Mason, Starke, Komura SCA 2022",
    "aistpp":   "CC-BY 4.0 annotations — Li et al. ICCV 2021",
    "mhad":     "BSD-2-Clause — Ofli et al. 2013",
}


@dataclass
class MotionClip:
    source: str              # "cmu" / "100style" / "aistpp" / "mhad"
    path: Path
    license: str
    description: str = ""    # human-readable motion category

    @property
    def id(self) -> str:
        return f"{self.source}:{self.path.stem}"


def iter_cmu(root: Path, descriptions: dict[str, str] | None = None) -> Iterator[MotionClip]:
    for p in sorted(root.glob("*.bvh")):
        desc = (descriptions or {}).get(p.stem, "")
        yield MotionClip("cmu", p, SOURCE_LICENSE["cmu"], desc)


def iter_100style(root: Path) -> Iterator[MotionClip]:
    # 100STYLE layout: <root>/<StyleName>/<StyleName>_<MovementType>.bvh
    for style_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(style_dir.glob("*.bvh")):
            yield MotionClip("100style", p, SOURCE_LICENSE["100style"],
                              description=f"{style_dir.name}/{p.stem}")


def iter_aistpp(root: Path) -> Iterator[MotionClip]:
    # Converted AIST++ BVH — filename encodes genre_choreography_subject_music_tempo
    for p in sorted(root.glob("*.bvh")):
        yield MotionClip("aistpp", p, SOURCE_LICENSE["aistpp"], description="dance")


def iter_mhad(root: Path) -> Iterator[MotionClip]:
    for p in sorted(root.glob("*.bvh")):
        yield MotionClip("mhad", p, SOURCE_LICENSE["mhad"], description="")


def load_all_clips(assets_root: Path) -> list[MotionClip]:
    """Enumerate every motion clip across every present source."""
    clips: list[MotionClip] = []
    cmu_dir = assets_root / "bvh"
    if cmu_dir.exists():
        clips.extend(iter_cmu(cmu_dir))
    style_dir = assets_root / "bvh_100style" / "100STYLE"
    if style_dir.exists():
        clips.extend(iter_100style(style_dir))
    aist_dir = assets_root / "aist_plusplus" / "bvh"
    if aist_dir.exists():
        clips.extend(iter_aistpp(aist_dir))
    mhad_dir = assets_root / "mhad" / "bvh"
    if mhad_dir.exists():
        clips.extend(iter_mhad(mhad_dir))
    return clips


def license_manifest(clips: list[MotionClip]) -> dict:
    """Return a per-source summary suitable for a dataset card / ATTRIBUTION.md."""
    summary: dict = {}
    for c in clips:
        entry = summary.setdefault(c.source, {
            "license": c.license,
            "n_clips": 0,
            "examples": [],
        })
        entry["n_clips"] += 1
        if len(entry["examples"]) < 3:
            entry["examples"].append(c.path.name)
    return summary
