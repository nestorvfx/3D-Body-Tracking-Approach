"""Print source BVH arm/leg along directions across a range of frames.

Lets us see if the source has real arm swing, leg stride, etc., before
blaming the retargeter.
"""
import sys
from pathlib import Path
import bpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

def ensure_mpfb():
    try: bpy.ops.preferences.addon_enable(module="bl_ext.user_default.mpfb")
    except: pass

ensure_mpfb()
from lib.fk_retarget import load_bvh

args = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
bvh_path = args[0]
frames = [int(x) for x in args[1:]] if len(args) > 1 else [100, 142, 180, 220, 260]

bvh = load_bvh(bvh_path, source=None)
src_kind = bvh.get("_mocap_source", "?")
print(f"source={src_kind}, frames={frames}")

# Guess the arm/leg/root bones for each source
names_by_src = {
    "cmu":     ("Hips", "LeftArm", "LeftForeArm", "LeftUpLeg", "LeftLeg"),
    "100style": ("Hips", "LeftShoulder", "LeftElbow", "LeftHip", "LeftKnee"),
    "aistpp":  ("pelvis", "l_shoulder", "l_elbow", "l_hip", "l_knee"),
}
root, sh, elb, hip, knee = names_by_src.get(src_kind, names_by_src["cmu"])

print(f"\n{'frame':<6s} | {'root_world':>28s} | "
      f"{sh+'_along':>22s} | {elb+'_along':>22s} | "
      f"{hip+'_along':>22s} | {knee+'_along':>22s}")

src_mw = bvh.matrix_world
for f in frames:
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    src_eval = bvh.evaluated_get(dg)
    def b(name):
        pb = src_eval.pose.bones.get(name)
        if pb is None: return (None, None)
        m = src_mw @ pb.matrix
        return (m.translation, m.col[1].to_3d().normalized())
    rh, _ = b(root)
    _, sha = b(sh)
    _, elba = b(elb)
    _, hipa = b(hip)
    _, kna = b(knee)

    def fmt_v(v, nd=3):
        if v is None: return "--"
        if nd == 3:
            return f"({v.x:+6.3f},{v.y:+6.3f},{v.z:+6.3f})"
        return f"({v.x:+5.2f},{v.y:+5.2f},{v.z:+5.2f})"

    print(f"{f:<6d} | {fmt_v(rh):>28s} | "
          f"{fmt_v(sha, 2):>22s} | {fmt_v(elba, 2):>22s} | "
          f"{fmt_v(hipa, 2):>22s} | {fmt_v(kna, 2):>22s}")
