"""Draw 2D keypoints on sample images to verify the label pipeline.
Outputs <out>/verify/<id>_kps.png per sample inspected."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from_dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/output/synth_v1")
n_inspect = int(sys.argv[2]) if len(sys.argv) > 2 else 8

labels = []
with (from_dataset / "labels.jsonl").open() as fh:
    for line in fh:
        labels.append(json.loads(line))

out_dir = from_dataset / "verify"
out_dir.mkdir(exist_ok=True)

COCO_SKELETON = [
    (5,7),(7,9),(6,8),(8,10),(5,6),
    (5,11),(6,12),(11,12),(11,13),(13,15),
    (12,14),(14,16),(0,1),(1,3),(0,2),(2,4),
]
KP_COLOR = (255, 80, 80)
BONE_COLOR = (80, 200, 255)
BBOX_COLOR = (80, 255, 120)

for rec in labels[:n_inspect]:
    img_path = from_dataset / rec["image_rel"]
    if not img_path.exists():
        continue
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    # bbox
    bx, by, bw, bh = rec["bbox_xywh"]
    draw.rectangle([(bx, by), (bx+bw, by+bh)], outline=BBOX_COLOR, width=2)
    kps = rec["keypoints_2d"]
    # bones
    for a, b in COCO_SKELETON:
        ka, kb = kps[a], kps[b]
        if ka[2] > 0 and kb[2] > 0:
            draw.line([(ka[0], ka[1]), (kb[0], kb[1])],
                      fill=BONE_COLOR, width=2)
    # keypoints
    for u, v, vis in kps:
        if vis > 0:
            r = 3
            draw.ellipse([(u-r, v-r), (u+r, v+r)], fill=KP_COLOR,
                         outline=(255, 255, 255))
    out_path = out_dir / f"{rec['id']}_kps.png"
    img.save(out_path)
    print(f"  verified {out_path.name}")

print(f"[done] wrote {n_inspect} verification images to {out_dir}")
