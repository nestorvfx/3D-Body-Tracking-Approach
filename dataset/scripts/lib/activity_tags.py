"""Activity category tagging for motion clips across sources.

Goal: group clips into canonical categories so the sampler can enforce
per-category frame budgets (avoid "all walking" dataset).

Categories:
  locomotion : walk, run, jog, skip, hop
  dance      : any dance style
  fitness    : stretching, exercise, calisthenics, yoga
  martial    : boxing, kicking, self-defence, karate
  sports     : basketball, soccer, baseball, golf
  gesture    : waving, pointing, communicative motion
  daily      : sit/stand, washing, drinking, eating, manipulation
  acrobatic  : jump, flip, cartwheel, balance, gymnastics
  idle       : standing still, minor movement
  unknown    : everything else
"""
from __future__ import annotations

from pathlib import Path
import re


CATEGORIES = ("locomotion", "dance", "fitness", "martial",
              "sports", "gesture", "daily", "acrobatic", "idle", "unknown")


# --- 100STYLE: 100 style folder names mapped to categories.  Most are
# locomotion variations, a few are dance/acrobatic.
_100STYLE_CATEGORIES = {
    "Akimbo": "locomotion", "Angry": "locomotion", "ArmsAboveHead": "locomotion",
    "ArmsBehindBack": "locomotion", "ArmsBySide": "locomotion",
    "ArmsFolded": "locomotion", "Balance": "acrobatic", "BeatChest": "gesture",
    "BentForward": "locomotion", "BentKnees": "locomotion",
    "BigSteps": "locomotion", "BouncyLeft": "locomotion",
    "BouncyRight": "locomotion", "Cat": "locomotion", "Chicken": "locomotion",
    "CrossOver": "locomotion", "Crouched": "locomotion",
    "CrowdAvoidance": "locomotion", "DragLeft": "locomotion",
    "DragRight": "locomotion", "Drunk": "locomotion", "DuckFoot": "locomotion",
    "Elated": "locomotion", "Exaggerated": "locomotion",
    "FairySteps": "dance", "FlappingArms": "gesture", "Followed": "locomotion",
    "Giant": "locomotion", "Graceful": "dance", "HandsBetweenLegs": "locomotion",
    "HandsInPockets": "locomotion", "HighKnees": "fitness",
    "Hobble": "locomotion", "HopSkipJump": "acrobatic", "Hurt": "locomotion",
    "InTheDark": "locomotion", "Joy": "locomotion", "Kick": "martial",
    "LawnMower": "locomotion", "LeanBack": "locomotion",
    "LeanLeft": "locomotion", "LeanRight": "locomotion",
    "LeftHop": "acrobatic", "LegsApart": "locomotion",
    "LimpLeft": "locomotion", "LimpRight": "locomotion",
    "Lunge": "fitness", "March": "locomotion", "Mask": "locomotion",
    "Mech": "locomotion", "Monk": "locomotion", "Morris": "dance",
    "Neutral": "locomotion", "NotHands": "locomotion", "Old": "locomotion",
    "OnHeels": "locomotion", "OnPhoneLeft": "daily", "OnPhoneRight": "daily",
    "OnTheMoon": "locomotion", "OnToes": "locomotion",
    "OnToesBentForward": "locomotion", "OnToesCrouched": "locomotion",
    "Penguin": "locomotion", "PrairieDog": "locomotion",
    "Pregnant": "locomotion", "Prepuber": "locomotion", "Quail": "locomotion",
    "RaisedLeftArm": "gesture", "RaisedRightArm": "gesture",
    "RightHop": "acrobatic", "Robot": "locomotion", "Roodle": "locomotion",
    "Rushed": "locomotion", "ShieldedLeft": "locomotion",
    "ShieldedRight": "locomotion", "Skip": "locomotion", "SlideFeet": "locomotion",
    "SpinAntiClock": "dance", "SpinClock": "dance", "Star": "acrobatic",
    "StartStop": "locomotion", "Stiff": "locomotion", "Strutting": "locomotion",
    "Superman": "acrobatic", "Swat": "martial", "Sweep": "daily",
    "Swimming": "sports", "SwingArmsRound": "gesture",
    "SwingShoulders": "gesture", "Teapot": "gesture", "Tiptoe": "locomotion",
    "TogetherStep": "locomotion", "TwoFootJump": "acrobatic",
    "WalkingStickLeft": "daily", "WalkingStickRight": "daily",
    "Waving": "gesture", "WhirlArms": "gesture", "WideLegs": "locomotion",
    "WiggleHips": "dance", "WildArms": "gesture", "WildLegs": "locomotion",
    "Zombie": "locomotion",
}


# --- CMU: keyword-based tagging from description strings in cmuindex.txt
_CMU_DESC_PATTERNS = [
    (r"\b(dance|ballet|salsa|samba|tango|waltz)\b", "dance"),
    (r"\b(walk|walking|stroll|march)\b", "locomotion"),
    (r"\b(run|running|jog|sprint)\b", "locomotion"),
    (r"\b(jump|leap|hop|vault|flip|cartwheel|somersault)\b", "acrobatic"),
    (r"\b(box|boxing|punch|kick|strike|martial|karate|fight)\b", "martial"),
    (r"\b(basketball|soccer|football|baseball|golf|tennis)\b", "sports"),
    (r"\b(yoga|stretch|warm|cooldown|exercise|squat|lunge)\b", "fitness"),
    (r"\b(sit|stand|rise|lie|lay|bend|reach|grab|drink|eat|clap|wave)\b", "daily"),
    (r"\b(gesture|point|signal|salute)\b", "gesture"),
    (r"\b(idle|stand still|rest)\b", "idle"),
]


def _tag_from_patterns(desc: str) -> str:
    if not desc:
        return "unknown"
    desc = desc.lower()
    for pattern, cat in _CMU_DESC_PATTERNS:
        if re.search(pattern, desc):
            return cat
    return "unknown"


def tag_100style(clip_path: Path) -> str:
    """100STYLE filename: <style_dir>/<style>_<suffix>.bvh"""
    style = clip_path.parent.name
    return _100STYLE_CATEGORIES.get(style, "locomotion")


def tag_cmu(clip_path: Path, descriptions: dict[str, str] | None = None) -> str:
    """CMU: tag from description mapping if available, else keyword from filename."""
    if descriptions:
        d = descriptions.get(clip_path.stem, "")
        if d:
            return _tag_from_patterns(d)
    return _tag_from_patterns(clip_path.stem.replace("_", " "))


def tag_aistpp(clip_path: Path) -> str:
    """AIST++ filenames are always dance."""
    return "dance"


def tag_mhad(clip_path: Path) -> str:
    """Berkeley MHAD actions map loosely to fitness/gesture/daily."""
    name = clip_path.stem.lower()
    if any(k in name for k in ("jump", "jack", "clap", "wave")):
        return "fitness" if "jack" in name else "gesture"
    if "throw" in name or "punch" in name:
        return "martial"
    if "sit" in name or "stand" in name:
        return "daily"
    return "daily"


def tag_clip(source: str, path: Path,
             descriptions: dict[str, str] | None = None) -> str:
    if source == "cmu":
        return tag_cmu(path, descriptions)
    if source == "100style":
        return tag_100style(path)
    if source == "aistpp":
        return tag_aistpp(path)
    if source == "mhad":
        return tag_mhad(path)
    return "unknown"


def parse_cmu_index(index_text: str) -> dict[str, str]:
    """Parse cmuindex.txt into {clip_stem: description} dict."""
    descs: dict[str, str] = {}
    for line in index_text.splitlines():
        m = re.match(r"^(\d{2,3}_\d{2,3})\s+(.+?)$", line.strip())
        if m:
            descs[m.group(1)] = m.group(2)
    return descs
