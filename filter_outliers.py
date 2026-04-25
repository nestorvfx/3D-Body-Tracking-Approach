import json, os, math
root = "dataset/output/synth_v3"
src = f"{root}/labels.jsonl"
dst = f"{root}/labels.jsonl.tmp"
MAX_ROOT_Z = 50.0
MIN_ROOT_Z = 0.3
kept = dropped = 0
with open(src) as fi, open(dst, "w") as fo:
    for line in fi:
        rec = json.loads(line)
        kps3d = rec["keypoints_3d_cam"]
        l, r = kps3d[11], kps3d[12]
        if l is None or r is None:
            dropped += 1; continue
        rz = (l[2] + r[2]) / 2
        if not (MIN_ROOT_Z <= abs(rz) <= MAX_ROOT_Z) or math.isnan(rz):
            dropped += 1; continue
        fo.write(line); kept += 1
os.replace(dst, src)
print(f"kept={kept} dropped={dropped}")
