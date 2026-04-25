"""QA overlay for pilot_v2 sequence outputs.

For each seq_XXXX/labels.json, draws 2D COCO-17 overlays on the rendered
frames and composites a contact sheet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {
    "visible": (0, 255, 0, 255),
    "offscreen": (255, 100, 100, 200),
    "edge_l": (80, 160, 255, 220),
    "edge_r": (255, 180, 80, 220),
    "edge_mid": (255, 255, 255, 220),
}


def side_for_edge(a, b):
    left = {1, 3, 5, 7, 9, 11, 13, 15}
    right = {2, 4, 6, 8, 10, 12, 14, 16}
    if a in left and b in left:
        return "edge_l"
    if a in right and b in right:
        return "edge_r"
    return "edge_mid"


def render_overlay(png_path, meta, out_path, edges):
    img = Image.open(png_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    kps = meta["keypoints_2d_pixels"]

    for a, b in edges:
        ka, kb = kps[a], kps[b]
        if not ka or not kb or ka["px"] is None or kb["px"] is None:
            continue
        c = COLORS[side_for_edge(a, b)]
        draw.line([(ka["px"], ka["py"]), (kb["px"], kb["py"])], fill=c, width=3)

    for i, kp in enumerate(kps):
        if kp is None or kp["px"] is None:
            continue
        c = COLORS["visible"] if kp["visible"] else COLORS["offscreen"]
        x, y = kp["px"], kp["py"]
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c, outline=(0, 0, 0, 255))
        draw.text((x + 6, y - 6), str(i), fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(img, overlay)
    combined.convert("RGB").save(out_path, "PNG")


def main():
    here = Path(__file__).resolve().parent
    pilot_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (here.parent / "output" / "pilot_v2_10")

    all_overlays = []
    seq_dirs = sorted(pilot_dir.glob("seq_*"))
    if not seq_dirs:
        print(f"No seq_* dirs in {pilot_dir}")
        return 1

    for seq_dir in seq_dirs:
        labels_path = seq_dir / "labels.json"
        if not labels_path.exists():
            continue
        labels = json.loads(labels_path.read_text())
        edges = labels["skeleton_edges"]
        overlay_dir = seq_dir / "overlays"
        overlay_dir.mkdir(exist_ok=True)

        for frame_ann in labels["frames"]:
            png = seq_dir / frame_ann["png_path"]
            if not png.exists():
                continue
            out = overlay_dir / f"overlay_{frame_ann['frame_index']:04d}.png"
            try:
                render_overlay(png, frame_ann, out, edges)
                all_overlays.append(out)
            except Exception as e:
                print(f"  {png.name}: {e}")

    if not all_overlays:
        print("No overlays produced")
        return 1

    # Contact sheet: arrange up to 40 overlays in a grid
    n = min(len(all_overlays), 40)
    sample = Image.open(all_overlays[0])
    w, h = sample.size
    thumb_w, thumb_h = w // 3, h // 3
    cols = 8
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (32, 32, 32))
    for i, p in enumerate(all_overlays[:n]):
        im = Image.open(p).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
        sheet.paste(im, ((i % cols) * thumb_w, (i // cols) * thumb_h))
    sheet_path = pilot_dir / "contact_sheet.png"
    sheet.save(sheet_path, "PNG", optimize=True)
    print(f"contact sheet: {sheet_path} ({n} frames)")
    print(f"overlays total: {len(all_overlays)} across {len(seq_dirs)} sequences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
