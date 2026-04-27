import sys
import cv2
import numpy as np
from albumentations.augmentations.mixing.domain_adaptation_functional import fourier_domain_adaptation
from PIL import Image, ImageDraw

img_bgr = cv2.imread(
    r"c:/Users/Mihajlo/Documents/Body Tracking/dataset/output/synth_iter/images/cmu_01_08_f0144_s42_c0.png")
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
ref_bgr = cv2.imread(
    r"c:/Users/Mihajlo/Documents/Body Tracking/assets/sim2real_refs/fda/frame_0000.png")
ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
print("img", img.shape, img.dtype, "ref", ref.shape, ref.dtype)

panels = [(img, "orig")]
for beta in [0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]:
    out = fourier_domain_adaptation(img, ref, beta)
    out_u = np.clip(out, 0, 255).astype(np.uint8)
    panels.append((out_u, f"beta={beta}"))
    print(f"beta={beta}, out range [{out.min():.1f}, {out.max():.1f}], "
          f"mean={out.mean():.1f}")

H = max(p.shape[0] for p, _ in panels)
W = sum(p.shape[1] + 8 for p, _ in panels) + 8
strip = Image.new("RGB", (W, H + 28), (20, 22, 26))
draw = ImageDraw.Draw(strip)
x = 4
for arr, lbl in panels:
    pil = Image.fromarray(arr)
    strip.paste(pil, (x, 28))
    draw.text((x + 4, 6), lbl, fill=(220, 220, 220))
    x += pil.width + 8
strip.save(r"c:/Users/Mihajlo/Documents/Body Tracking/dataset/output/aug_debug_v2/_fda_sweep.png")
print("wrote _fda_sweep.png")
