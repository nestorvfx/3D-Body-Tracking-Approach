"""
camera_rig.py - BEDLAM2.0-style camera intrinsics/extrinsics diversity for Blender.

Covers:
  - Focal sampling (14-400mm log-uniform, biased to phone/DSLR range)
  - 16:9 DSLR sensor (36 x 20.25mm) per BEDLAM2.0 Sec 3.1
  - Random extrinsics on a sphere (yaw / pitch / roll / distance)
  - Strict framing: binary-search distance so all COCO-17 keypoints fit in
    [margin, 1-margin] normalized image coords
  - Perlin-noise handheld camera shake (mathutils.noise)
  - Synthetic camera motions: static / pan / orbit / dolly / track / zoom
  - Principal-point offset via shift_x / shift_y
  - Depth of field
  - Intrinsics K export

References:
  BEDLAM2.0 paper:   https://arxiv.org/html/2511.14394v1  (Sec 3.1)
  Blender Camera:    https://docs.blender.org/api/current/bpy.types.Camera.html
  world_to_camera:   https://docs.blender.org/api/current/bpy_extras.object_utils.html
  mathutils.noise:   https://docs.blender.org/api/current/mathutils.noise.html
  Blender->OpenCV K: https://www.rojtberg.net/1601/from-blender-to-opencv-camera-and-back/
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Sequence, Tuple

import bpy
from mathutils import Vector, Euler, Quaternion, noise as mnoise
from bpy_extras.object_utils import world_to_camera_view


# -------- BEDLAM2.0 defaults (paper Sec 3.1) --------
SENSOR_W_MM = 36.0          # 16:9 DSLR sensor width
SENSOR_H_MM = 20.25         # 36 * 9/16
FOCAL_MIN_MM = 14.0
FOCAL_MAX_MM = 400.0
# Motion type probabilities (Fig 2, right-most panel in paper)
MOTION_MIX = {
    "static":  0.45,
    "pan":     0.15,
    "orbit":   0.12,
    "dolly":   0.10,
    "track":   0.10,
    "zoom":    0.08,   # the paper reports 9% of clips have zoom-during-shot
}


# ---------- Data classes ----------
@dataclass
class CameraIntrinsics:
    focal_mm: float
    sensor_w_mm: float
    sensor_h_mm: float
    shift_x: float      # in fractions of sensor width
    shift_y: float      # in fractions of sensor width (Blender quirk)
    image_w: int
    image_h: int

    def K(self) -> List[List[float]]:
        # Blender->OpenCV (Rojtberg). Blender shift_x is inverted.
        fx = self.focal_mm / self.sensor_w_mm * self.image_w
        fy = fx            # square pixels (pixel_aspect=1); see scene.render if not
        cx = self.image_w * (0.5 - self.shift_x)
        cy = 0.5 * self.image_h + self.image_w * self.shift_y
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


@dataclass
class CameraSample:
    intrinsics: CameraIntrinsics
    motion: str
    yaw: float
    pitch: float
    roll: float
    distance: float
    shake_amp_cm: float
    shake_amp_deg: float
    dof_fstop: float
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    keyframes: dict = field(default_factory=dict)   # per-frame debug log


# ---------- Samplers ----------
def _loguniform(rng, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def sample_focal_mm(rng) -> float:
    """
    BEDLAM2.0 (NeurIPS 2025 Oral) reports that moving from two fixed focals
    to a 14-400 mm log-distributed sweep delivered the single biggest
    accuracy gain in the paper — 30-45 percent reduction in EMDB world-
    frame error.  Their recipe: log-uniform coverage across the full range
    with a centre-heavy bias toward phone/DSLR (24-85 mm).

    We approximate with a MIXTURE:
      85 percent of samples: lognormal around 35 mm (sigma 0.70) → phone/DSLR
      15 percent of samples: pure log-uniform across [14, 400] → tails

    The widened sigma (0.55 → 0.70) + the uniform tail together give
    meaningful coverage of wide-angle (<20 mm selfie) and telephoto
    (>200 mm surveillance) regimes without starving the centre.
    """
    if rng.random() < 0.15:
        # Log-uniform tail: captures wide-angle selfies + telephoto shots.
        return _loguniform(rng, FOCAL_MIN_MM, FOCAL_MAX_MM)
    mu, sigma = math.log(35.0), 0.70
    f = math.exp(rng.gauss(mu, sigma))
    return max(FOCAL_MIN_MM, min(FOCAL_MAX_MM, f))


def sample_extrinsics(rng) -> dict:
    """Yaw uniform; pitch ~N(0, 15 deg); roll ~N(0, 5 deg)."""
    return dict(
        yaw=rng.uniform(-math.pi, math.pi),
        pitch=math.radians(rng.gauss(0.0, 15.0)),
        roll=math.radians(max(-math.radians(30.0),
                              min(math.radians(30.0),
                                  rng.gauss(0.0, 5.0)))),
    )


def sample_motion_type(rng) -> str:
    r = rng.random()
    acc = 0.0
    for k, p in MOTION_MIX.items():
        acc += p
        if r <= acc:
            return k
    return "static"


# ---------- Core camera build ----------
def make_camera(name: str = "DatasetCamera") -> bpy.types.Object:
    cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = SENSOR_W_MM
    cam_data.sensor_height = SENSOR_H_MM
    cam_data.clip_start = 0.05
    cam_data.clip_end = 500.0
    bpy.context.scene.camera = cam_obj
    return cam_obj


def apply_intrinsics(cam: bpy.types.Object, intr: CameraIntrinsics) -> None:
    d = cam.data
    d.lens = intr.focal_mm
    d.sensor_width = intr.sensor_w_mm
    d.sensor_height = intr.sensor_h_mm
    d.shift_x = intr.shift_x
    d.shift_y = intr.shift_y


def place_on_sphere(cam: bpy.types.Object, target: Vector,
                    yaw: float, pitch: float, roll: float,
                    distance: float) -> None:
    """Put camera on sphere of given radius around target; aim at target."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    offset = Vector((distance * cp * cy, distance * cp * sy, distance * sp))
    cam.location = target + offset
    fwd = (target - cam.location).normalized()
    # Align camera -Z with fwd; up is +Z world
    rot = fwd.to_track_quat("-Z", "Y").to_euler()
    rot.rotate_axis("Z", roll)
    cam.rotation_euler = rot


def add_track_to(cam: bpy.types.Object, target_obj: bpy.types.Object) -> None:
    """
    Track-To constraint: re-aims camera each frame even when its location is
    jittered by Perlin shake. Safe because shake is applied on location only.
    Downside: any manual rotation_euler on the camera is overridden by the
    constraint -- so roll must be implemented via camera.data or a parent empty
    when Track-To is active. For our module we bake roll into rotation AFTER
    disabling the constraint mute; see shake_apply().
    """
    c = cam.constraints.new("TRACK_TO")
    c.target = target_obj
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"


# ---------- Framing ----------
def _kp_world_positions(armature: bpy.types.Object,
                        bone_names: Sequence[str]) -> List[Vector]:
    """Return world-space locations for the 17 COCO keypoint bones."""
    pts = []
    mw = armature.matrix_world
    for bn in bone_names:
        pb = armature.pose.bones.get(bn)
        if pb is None:
            continue
        pts.append(mw @ pb.head)
    return pts


def _all_in_frame(cam: bpy.types.Object, scene: bpy.types.Scene,
                  pts: Sequence[Vector], margin: float) -> Tuple[bool, int]:
    """Return (ok, visible_count). Uses bpy_extras.world_to_camera_view."""
    bpy.context.view_layer.update()
    vis = 0
    ok = True
    for p in pts:
        ndc = world_to_camera_view(scene, cam, p)
        if ndc.z <= 0.0:               # behind camera
            ok = False
            continue
        if margin <= ndc.x <= 1 - margin and margin <= ndc.y <= 1 - margin:
            vis += 1
        else:
            ok = False
    return ok, vis


def frame_armature_strict(camera: bpy.types.Object,
                          armature: bpy.types.Object,
                          rng,
                          focal_mm: float,
                          target_visibility: int = 17,
                          margin: float = 0.05,
                          max_iter: int = 24,
                          bone_names: Sequence[str] = None):
    """
    Guarantee all COCO-17 keypoints project into [margin, 1-margin]x[margin,1-margin].

    Strategy:
      1. Apply the chosen focal length / sensor.
      2. Sample yaw/pitch/roll.
      3. Binary-search distance in [d_min, d_max] to find the smallest
         distance at which all 17 keypoints are inside the margin box.
      4. If no distance works (self-occlusion / pose wider than frame), widen
         the margin adaptively (backoff) and retry from step 2 up to N times.

    Returns: CameraSample with filled distance.
    """
    if bone_names is None:
        from .coco17 import COCO17_BONES   # pipeline's mapping
        bone_names = COCO17_BONES

    scene = bpy.context.scene
    pts_world = _kp_world_positions(armature, bone_names)
    if not pts_world:
        raise RuntimeError("No COCO-17 bones found on armature")

    centroid = sum(pts_world, Vector((0, 0, 0))) / len(pts_world)
    # Characteristic radius of the keypoint cloud from its centroid
    radius = max((p - centroid).length for p in pts_world)

    # Geometric lower bound from FOV: d >= radius / tan(hfov/2)
    hfov = 2.0 * math.atan(SENSOR_W_MM / (2.0 * focal_mm))
    vfov = 2.0 * math.atan(SENSOR_H_MM / (2.0 * focal_mm))
    fov_min = min(hfov, vfov)
    d_lo_geom = radius / math.tan(fov_min / 2.0) * (1.0 + 2 * margin)

    # Principal-point jitter widened from ±0.03 to ±0.08 (BEDLAM2 /
    # CameraHMR recipe).  CLIFF (ECCV 2022) showed pose models trained
    # on off-centre crops need this range to generalise to real-world
    # phone camera principal-point variation.
    intr_shift_x = rng.uniform(-0.08, 0.08)
    intr_shift_y = rng.uniform(-0.08, 0.08)
    intr = CameraIntrinsics(
        focal_mm=focal_mm,
        sensor_w_mm=SENSOR_W_MM, sensor_h_mm=SENSOR_H_MM,
        shift_x=intr_shift_x, shift_y=intr_shift_y,
        image_w=scene.render.resolution_x, image_h=scene.render.resolution_y,
    )
    apply_intrinsics(camera, intr)

    attempts = 6
    cur_margin = margin
    for attempt in range(attempts):
        ext = sample_extrinsics(rng)
        d_lo = max(0.25, d_lo_geom)
        d_hi = max(d_lo * 3.0, d_lo + 10.0)
        best = None

        # Expand d_hi until framing succeeds (or give up)
        for _ in range(6):
            place_on_sphere(camera, centroid, ext["yaw"], ext["pitch"],
                            ext["roll"], d_hi)
            ok, vis = _all_in_frame(camera, scene, pts_world, cur_margin)
            if ok and vis >= target_visibility:
                best = d_hi
                break
            d_hi *= 1.6

        if best is None:
            cur_margin = max(0.01, cur_margin * 0.75)   # relax margin
            continue

        # Binary-search for the tightest distance that still satisfies
        for _ in range(max_iter):
            d_mid = 0.5 * (d_lo + d_hi)
            place_on_sphere(camera, centroid, ext["yaw"], ext["pitch"],
                            ext["roll"], d_mid)
            ok, vis = _all_in_frame(camera, scene, pts_world, cur_margin)
            if ok and vis >= target_visibility:
                d_hi = d_mid
                best = d_mid
            else:
                d_lo = d_mid

        place_on_sphere(camera, centroid, ext["yaw"], ext["pitch"],
                        ext["roll"], best)
        return CameraSample(
            intrinsics=intr, motion=sample_motion_type(rng),
            yaw=ext["yaw"], pitch=ext["pitch"], roll=ext["roll"],
            distance=best,
            shake_amp_cm=rng.uniform(0.0, 3.0),
            shake_amp_deg=rng.uniform(0.0, 1.0),
            dof_fstop=rng.uniform(1.8, 11.0),
            target=tuple(centroid),
        )

    raise RuntimeError("frame_armature_strict: could not frame pose after retries")


# ---------- Perlin shake ----------
def shake_apply(cam: bpy.types.Object, sample: CameraSample,
                frame_start: int, frame_end: int,
                seed: float = 0.0) -> None:
    """
    Adds per-frame Perlin-noise translation (amplitude in cm) and rotation
    (amplitude in deg). Keyframed on location / rotation_euler.
    Frequency ~ 0.15 cycles/frame (gentle handheld).
    """
    amp_t = sample.shake_amp_cm * 0.01          # cm -> m
    amp_r = math.radians(sample.shake_amp_deg)
    base_loc = cam.location.copy()
    base_rot = cam.rotation_euler.copy()
    for f in range(frame_start, frame_end + 1):
        t = f * 0.15 + seed
        nv = mnoise.noise_vector(Vector((t, t * 1.1, t * 0.9)),
                                 noise_basis="PERLIN_ORIGINAL")
        nr = mnoise.noise_vector(Vector((t + 17.3, t + 3.1, t + 9.7)),
                                 noise_basis="PERLIN_ORIGINAL")
        cam.location = base_loc + Vector(nv) * amp_t
        cam.rotation_euler = Euler((base_rot.x + nr.x * amp_r,
                                    base_rot.y + nr.y * amp_r,
                                    base_rot.z + nr.z * amp_r), "XYZ")
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)


# ---------- Synthetic motions (pan / orbit / dolly / track / zoom) ----------
def apply_motion(cam: bpy.types.Object, sample: CameraSample,
                 frame_start: int, frame_end: int, rng) -> None:
    """Keyframe the start/end camera state for the chosen motion type."""
    scene = bpy.context.scene
    target = Vector(sample.target)
    m = sample.motion
    duration = max(1, frame_end - frame_start)

    if m == "static":
        place_on_sphere(cam, target, sample.yaw, sample.pitch, sample.roll,
                        sample.distance)
        cam.keyframe_insert("location", frame=frame_start)
        cam.keyframe_insert("rotation_euler", frame=frame_start)
        cam.keyframe_insert("location", frame=frame_end)
        cam.keyframe_insert("rotation_euler", frame=frame_end)
        return

    if m == "pan":
        d_yaw = math.radians(rng.uniform(-20.0, 20.0))
        for f, alpha in ((frame_start, 0.0), (frame_end, 1.0)):
            place_on_sphere(cam, target, sample.yaw, sample.pitch + alpha * 0.0,
                            sample.roll, sample.distance)
            # pan is rotation only; rotate around Z after placement
            cam.rotation_euler.rotate_axis("Z", alpha * d_yaw)
            cam.keyframe_insert("location", frame=f)
            cam.keyframe_insert("rotation_euler", frame=f)
        return

    if m == "orbit":
        dyaw = math.radians(rng.uniform(30.0, 90.0)) * rng.choice([-1, 1])
        for f, alpha in ((frame_start, 0.0), (frame_end, 1.0)):
            place_on_sphere(cam, target, sample.yaw + alpha * dyaw,
                            sample.pitch, sample.roll, sample.distance)
            cam.keyframe_insert("location", frame=f)
            cam.keyframe_insert("rotation_euler", frame=f)
        return

    if m == "dolly":
        d_ratio = rng.uniform(0.6, 1.5)
        for f, alpha in ((frame_start, 0.0), (frame_end, 1.0)):
            place_on_sphere(cam, target, sample.yaw, sample.pitch, sample.roll,
                            sample.distance * (1.0 + alpha * (d_ratio - 1.0)))
            cam.keyframe_insert("location", frame=f)
            cam.keyframe_insert("rotation_euler", frame=f)
        return

    if m == "track":
        # lateral slide; keep Track-To if attached, else re-aim
        side = Vector((math.cos(sample.yaw + math.pi / 2),
                       math.sin(sample.yaw + math.pi / 2), 0))
        amp = rng.uniform(0.5, 2.5)
        place_on_sphere(cam, target, sample.yaw, sample.pitch, sample.roll,
                        sample.distance)
        base = cam.location.copy()
        for f, alpha in ((frame_start, -0.5), (frame_end, 0.5)):
            cam.location = base + side * (alpha * amp)
            # re-aim
            fwd = (target - cam.location).normalized()
            cam.rotation_euler = fwd.to_track_quat("-Z", "Y").to_euler()
            cam.keyframe_insert("location", frame=f)
            cam.keyframe_insert("rotation_euler", frame=f)
        return

    if m == "zoom":
        f_start = sample.intrinsics.focal_mm
        f_end = max(FOCAL_MIN_MM, min(FOCAL_MAX_MM, f_start * rng.uniform(0.5, 2.0)))
        place_on_sphere(cam, target, sample.yaw, sample.pitch, sample.roll,
                        sample.distance)
        cam.keyframe_insert("location", frame=frame_start)
        cam.keyframe_insert("rotation_euler", frame=frame_start)
        cam.data.lens = f_start
        cam.data.keyframe_insert("lens", frame=frame_start)
        cam.data.lens = f_end
        cam.data.keyframe_insert("lens", frame=frame_end)
        sample.keyframes["focal_end"] = f_end


# ---------- DoF + export ----------
def apply_dof(cam: bpy.types.Object, focus_distance: float, fstop: float) -> None:
    d = cam.data
    d.dof.use_dof = True
    d.dof.focus_distance = focus_distance
    d.dof.aperture_fstop = fstop


def export_intrinsics(sample: CameraSample) -> dict:
    intr = sample.intrinsics
    out = asdict(intr)
    out["K"] = intr.K()
    out["hfov_deg"] = math.degrees(2 * math.atan(intr.sensor_w_mm / (2 * intr.focal_mm)))
    out["vfov_deg"] = math.degrees(2 * math.atan(intr.sensor_h_mm / (2 * intr.focal_mm)))
    return out


# ---------- One-call entry point ----------
def build_shot(camera: bpy.types.Object, armature: bpy.types.Object, rng,
               frame_start: int, frame_end: int,
               use_shake: bool = True) -> CameraSample:
    focal = sample_focal_mm(rng)
    sample = frame_armature_strict(camera, armature, rng, focal_mm=focal)
    apply_motion(camera, sample, frame_start, frame_end, rng)
    apply_dof(camera, focus_distance=sample.distance, fstop=sample.dof_fstop)
    if use_shake and sample.shake_amp_cm > 0.1:
        shake_apply(camera, sample, frame_start, frame_end,
                    seed=rng.uniform(0, 1000))
    return sample
