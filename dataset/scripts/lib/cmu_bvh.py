"""CMU BVH (cgspeed daz format) → MPFB2 "default" rig bone mapping.

CMU hierarchy (from 02_01.bvh header):
  Hips
   ├─ LHipJoint → LeftUpLeg → LeftLeg → LeftFoot → LeftToeBase
   ├─ RHipJoint → RightUpLeg → RightLeg → RightFoot → RightToeBase
   └─ LowerBack → Spine → Spine1 → {Neck → Neck1 → Head,
                                    LeftShoulder → LeftArm → LeftForeArm → LeftHand → ...,
                                    RightShoulder → RightArm → RightForeArm → RightHand → ...}

MPFB "default" rig uses the MakeHuman naming with .L/.R suffixes and numbered spine.
"""

# Only the bones that have clean 1:1 semantics. Helper/intermediate bones like
# LHipJoint, RHipJoint, LowerBack (CMU) and the MPFB twist bones (upperarm02,
# upperleg02, lowerarm02, lowerleg02, shoulder01) are left unconstrained so the
# rig's rest anatomy governs them.
CMU_TO_MPFB_DEFAULT = {
    # Root / translation
    "Hips":          "root",

    # Spine
    "Spine":         "spine03",
    "Spine1":        "spine01",

    # Neck / head
    "Neck":          "neck02",
    "Neck1":         "neck01",
    "Head":          "head",

    # Left arm
    "LeftShoulder":  "clavicle.L",
    "LeftArm":       "upperarm01.L",
    "LeftForeArm":   "lowerarm01.L",
    "LeftHand":      "wrist.L",

    # Right arm
    "RightShoulder": "clavicle.R",
    "RightArm":      "upperarm01.R",
    "RightForeArm":  "lowerarm01.R",
    "RightHand":     "wrist.R",

    # Left leg
    "LeftUpLeg":     "upperleg01.L",
    "LeftLeg":       "lowerleg01.L",
    "LeftFoot":      "foot.L",
    "LeftToeBase":   "toe1-1.L",

    # Right leg
    "RightUpLeg":    "upperleg01.R",
    "RightLeg":      "lowerleg01.R",
    "RightFoot":     "foot.R",
    "RightToeBase":  "toe1-1.R",
}

# cgspeed "daz" BVH reports offsets where one segment ≈ 2.5 units. Empirically
# a leg is 15–20 units, making a full body ≈ 60–80 units. Dividing by ~60 gives
# metres; 0.0178 is close to 1/56 and lines up characters ~1.8 m tall with
# MPFB's default 1.8 m mesh when HumanService.create_human is called at scale=1.0.
# The BVH importer's `global_scale` multiplies all translations + offsets, so
# using 0.0178 places the imported armature in metre space.
CMU_BVH_GLOBAL_SCALE = 0.0178
