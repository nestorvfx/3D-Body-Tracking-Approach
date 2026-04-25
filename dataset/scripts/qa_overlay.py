"""Draw 2D COCO-17 keypoints + skeleton onto the pilot PNGs.

Runs with Blender's bundled Python (has Pillow in Blender 5.x) or system Python.
Usage:
  python dataset/scripts/qa_overlay.py [pilot_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow not available. Install with: pip install pillow")
    sys.exit(2)


COLORS = {
    "visible": (0, 255, 0, 255),
    "offscreen": (255, 100, 100, 200),
    "edge_l": (80, 160, 255, 220),
    "edge_r": (255, 180, 80, 220),
    "edge_mid": (255, 255, 255, 220),
}


def side_for_edge(a: int, b: int) -> str:
    left = {1, 3, 5, 7, 9, 11, 13, 15}
    right = {2, 4, 6, 8, 10, 12, 14, 16}
    if a in left and b in left:
        return "edge_l"
    if a in right and b in right:
        return "edge_r"
    return "edge_mid"


def render_overlay(png_path: Path, meta: dict, out_path: Path) -> None:
    img = Image.open(png_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    kps = meta["keypoints_2d_pixels"]
    edges = meta["skeleton_edges"]

    # Draw skeleton edges
    for a, b in edges:
        ka, kb = kps[a], kps[b]
        if not ka or not kb:
            continue
        if ka["px"] is None or kb["px"] is None:
            continue
        color = COLORS[side_for_edge(a, b)]
        draw.line([(ka["px"], ka["py"]), (kb["px"], kb["py"])], fill=color, width=3)

    # Draw joints
    for i, kp in enumerate(kps):
        if kp is None or kp["px"] is None:
            continue
        color = COLORS["visible"] if kp["visible"] else COLORS["offscreen"]
        x, y = kp["px"], kp["py"]
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0, 255))
        draw.text((x + 6, y - 6), str(i), fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(img, overlay)
    combined.convert("RGB").save(out_path, "PNG")


def main() -> int:
    here = Path(__file__).resolve().parent
    pilot_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (here.parent / "output" / "pilot_10")
    summary_path = pilot_dir / "pilot_summary.json"
    if not summary_path.exists():
        print(f"No summary at {summary_path}")
        return 1
    summary = json.loads(summary_path.read_text())
    overlay_dir = pilot_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    n_ok = 0
    for m in summary["samples"]:
        png = pilot_dir / m["png"]
        if not png.exists():
            print(f"missing {png}")
            continue
        out = overlay_dir / f"overlay_{m['sample_id']:04d}.png"
        try:
            render_overlay(png, m, out)
            n_ok += 1
        except Exception as e:
            print(f"sample {m['sample_id']}: {e}")

    # Build a 2x5 contact sheet for quick visual sweep
    pngs = sorted(overlay_dir.glob("overlay_*.png"))
    if pngs:
        sample_img = Image.open(pngs[0])
        w, h = sample_img.size
        thumb_w, thumb_h = w // 2, h // 2
        cols, rows = 5, 2
        sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (32, 32, 32))
        for i, p in enumerate(pngs[:cols * rows]):
            im = Image.open(p).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
            sheet.paste(im, ((i % cols) * thumb_w, (i // cols) * thumb_h))
        sheet_path = pilot_dir / "contact_sheet.png"
        sheet.save(sheet_path, "PNG", optimize=True)
        print(f"contact sheet: {sheet_path}")

    print(f"overlays: {n_ok}/{len(summary['samples'])} -> {overlay_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
