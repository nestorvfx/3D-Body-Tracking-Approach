"""Quaternion swing-twist decomposition for drift-free retargeting.

Problem: MPFB's `default` rig has subdivided bones (`upperarm01.L`, `upperarm02.L`,
`upperarm03.L`) for anatomical twist.  CMU BVH only has a single `LeftArm`
bone.  Copy Rotation LOCAL-LOCAL leaves the twist sub-bones unconstrained, so
they keep their rest pose while the parent rotates — visibly shearing the arm.

Solution (Aberman et al., Skeleton-Aware Networks, SIGGRAPH 2020;
https://allenchou.net/2018/05/game-math-swing-twist-interpolation-sterp/):
split the source rotation into a swing component (perpendicular to the bone's
length axis) and a twist component (about the length axis).  Apply swing to
the primary bone, distribute the twist proportionally across the subdivided
chain.  This mirrors how human anatomy actually propagates long-axis rotation
along the limb.

Usage:
    decomp = SwingTwist(axis=Vector((0, 1, 0)))   # local Y = bone length
    q_swing, q_twist = decomp.decompose(q_source)
    # Distribute twist across e.g. [upperarm01, upperarm02, upperarm03]:
    for bone, w in zip(chain, [0.0, 0.4, 0.6]):
        bone.rotation_quaternion = q_swing * slerp(Quaternion(), q_twist, w)
"""
from __future__ import annotations

from dataclasses import dataclass

import mathutils  # type: ignore — Blender bundled


@dataclass
class SwingTwist:
    """Decompose a quaternion around a fixed local axis.

    `axis` must be a unit vector expressed in the bone's local rest frame
    (for Blender pose bones with default roll, this is Vector((0, 1, 0)) —
    the bone's local Y).
    """
    axis: "mathutils.Vector" = None     # type: ignore

    def __post_init__(self) -> None:
        if self.axis is None:
            self.axis = mathutils.Vector((0.0, 1.0, 0.0))
        self.axis = self.axis.normalized()

    def decompose(self, q: "mathutils.Quaternion") -> tuple["mathutils.Quaternion", "mathutils.Quaternion"]:
        """Return (swing, twist) such that `q == swing @ twist`.

        Twist is the rotation component about `self.axis`; swing is what's left.
        """
        q = q.normalized()
        # Project the vector part (x, y, z) onto the twist axis.
        v = mathutils.Vector((q.x, q.y, q.z))
        p = v.project(self.axis)
        # Twist quaternion has only the axis-aligned component of v, same w.
        twist = mathutils.Quaternion((q.w, p.x, p.y, p.z))
        if twist.magnitude < 1e-8:
            # q's vector part is perpendicular to axis → pure swing
            twist = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        else:
            twist.normalize()
        # swing = q * twist^-1  (conjugate of a unit quaternion is its inverse)
        swing = q @ twist.conjugated()
        swing.normalize()
        return swing, twist


def distribute_twist(
    q_twist: "mathutils.Quaternion",
    n: int,
    weights: list[float] | None = None,
) -> list["mathutils.Quaternion"]:
    """Split a twist quaternion into N fractional rotations summing to q_twist.

    Default weight schedule for 3-bone chains: [0.0, 0.4, 0.6] — shoulder/hip
    stays un-twisted, mid-twist carries 40%, elbow/knee-adjacent twist carries
    60% (matches anatomical reality; shoulder joints twist much less than
    the forearm/lower-leg).
    """
    if weights is None:
        if n == 3:
            weights = [0.0, 0.4, 0.6]
        elif n == 2:
            weights = [0.3, 0.7]
        else:
            weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"weights must have length {n}")
    identity = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    return [identity.slerp(q_twist, w) for w in weights]


def decompose_and_distribute(
    q_source: "mathutils.Quaternion",
    n_chain: int = 3,
    weights: list[float] | None = None,
    axis: "mathutils.Vector" = None,       # type: ignore
) -> tuple["mathutils.Quaternion", list["mathutils.Quaternion"]]:
    """Convenience: decompose q_source and return (swing, [twist_i, ...])
    where each twist_i is the fractional twist for bone i in the chain."""
    decomp = SwingTwist(axis=axis)
    swing, twist = decomp.decompose(q_source)
    twist_fractions = distribute_twist(twist, n_chain, weights)
    return swing, twist_fractions


def blend_swing_twist(
    swing: "mathutils.Quaternion",
    twist_fraction: "mathutils.Quaternion",
) -> "mathutils.Quaternion":
    """Recombine: primary bone gets `swing @ twist_fraction[0]`; each
    sub-bone gets just its `twist_fraction[i]` applied in its local frame."""
    return (swing @ twist_fraction).normalized()
