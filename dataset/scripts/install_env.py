"""One-shot environment setup: install MPFB2 extension + extract CC0 system asset pack.

Usage (from repo root, on Windows):
  "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" \
      --background --python dataset/scripts/install_env.py
"""
import os
import sys
import shutil
import zipfile
from pathlib import Path

import bpy  # type: ignore

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                                   # dataset/
MPFB_ZIP = REPO / "assets" / "addons" / "mpfb-2.0.14.zip"
ASSET_ZIP = REPO / "assets" / "mpfb_packs" / "makehuman_system_assets_cc0.zip"


def info(msg: str) -> None:
    print(f"[install_env] {msg}", flush=True)


def install_mpfb() -> None:
    info(f"Installing MPFB extension from {MPFB_ZIP}")
    if not MPFB_ZIP.exists():
        raise FileNotFoundError(MPFB_ZIP)
    # Sync user_default repo metadata first (creates the repo folder if missing).
    try:
        bpy.ops.preferences.addon_refresh()
    except Exception as e:
        info(f"(addon_refresh ignored: {e})")
    try:
        bpy.ops.extensions.package_install_files(
            filepath=str(MPFB_ZIP),
            repo="user_default",
            enable_on_install=True,
        )
        info("extensions.package_install_files OK")
    except Exception as e:
        info(f"package_install_files failed ({e}); trying legacy install_zip")
        # Fallback for older/newer Blender where the op name differs.
        bpy.ops.preferences.addon_install(
            filepath=str(MPFB_ZIP),
            overwrite=True,
        )
        bpy.ops.preferences.addon_enable(module="mpfb")
    bpy.ops.wm.save_userpref()
    info("Preferences saved.")


def locate_user_data() -> Path:
    """Return MPFB's user_data dir (where asset packs live)."""
    # Prefer the official API if available.
    try:
        from mpfb.services.locationservice import LocationService  # type: ignore
        return Path(LocationService.get_user_data())
    except Exception:
        pass
    # Fallback: replicate LocationService._user_home logic.
    # Blender 5.1 extension data path:
    appdata = os.environ.get("APPDATA") or str(Path.home() / ".config")
    major = f"{bpy.app.version[0]}.{bpy.app.version[1]}"
    base = Path(appdata) / "Blender Foundation" / "Blender" / major
    # Typical extension user data layout:
    #   <base>/extensions/user_default/mpfb/data/
    return base / "extensions" / "user_default" / "mpfb" / "data"


def extract_asset_pack() -> None:
    target = locate_user_data()
    target.mkdir(parents=True, exist_ok=True)
    info(f"Extracting {ASSET_ZIP.name} → {target}")
    if not ASSET_ZIP.exists():
        raise FileNotFoundError(ASSET_ZIP)
    with zipfile.ZipFile(ASSET_ZIP) as z:
        z.extractall(target)
    # Quick sanity count
    n = sum(1 for _ in target.rglob("*.mhclo"))
    m = sum(1 for _ in target.rglob("*.mhmat"))
    info(f"Extracted {n} .mhclo and {m} .mhmat files.")


def main() -> int:
    try:
        install_mpfb()
    except Exception as e:
        info(f"MPFB install failed: {e}")
        return 1
    try:
        extract_asset_pack()
    except Exception as e:
        info(f"Asset extraction failed: {e}")
        return 2
    info("OK — MPFB ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
