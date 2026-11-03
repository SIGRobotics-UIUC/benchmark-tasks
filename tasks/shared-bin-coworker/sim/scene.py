"""The rig: two YAM Pros side by side on one edge of a table, both reaching across it.

    y
    ^   +-----------------------+
    |   |[L] ->                 |       Seen from above.  The world frame is the real rig's (config/rig.yaml in
    0 --|                       |--> x  ~/Projects/score): origin midway between the bases, on the floor; x forward,
    |   |[R] ->                 |       y toward the left arm.  ``TABLE_TOP`` is the table's top.
        +-----------------------+

``python -m sim.reach`` gives the arm's reach; ``python -m sim.check`` tests the rig against it.

* Base separation, 0.50 m: the real rig's.
* Square, 0.90 m a side.  The bases sit flush with the back edge: their back faces, 35 mm behind the base axes, are on it.
* The far edge is 0.865 m from the bases and the arm's whole reach is 0.78 m, so nothing gets to it: the far part of
  the table is out of reach, and things go off the table over the left, right and back edges only.
* Height, 0.75 m, over a floor plane: things that go over the edge fall and stay fallen.
"""

from __future__ import annotations

import math

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from sim.robot import get_yam_cfg

TABLE_TOP = 0.75
TABLE_SIZE = (0.90, 0.90)
"""x (front to back) by y (side to side), metres."""
TABLE_TOP_MASS = 15.0
BASE_HALF_DEPTH = 0.035
"""From a base's axis to its back face."""
TABLE_CENTER_X = TABLE_SIZE[0] / 2 - BASE_HALF_DEPTH
"""Puts the back edge on the bases' back faces."""
TABLE_THICKNESS = 0.03
LEG_INSET = 0.10
"""Legs stand this far in from the edges, so anything going over an edge reaches the floor."""
BASE_Y = 0.25
"""Each arm's base is this far from the centre along y."""

LEFT_ARM, RIGHT_ARM = "left_arm", "right_arm"
_SHELL_BITS = {LEFT_ARM: (2, 4), RIGHT_ARM: (4, 2)}
BASE_POSE = {
  LEFT_ARM: ((0.0, BASE_Y, TABLE_TOP), (1.0, 0.0, 0.0, 0.0)),
  RIGHT_ARM: ((0.0, -BASE_Y, TABLE_TOP), (1.0, 0.0, 0.0, 0.0)),
}

CUBE_HALF = 0.02
CUBE_MASS = 0.05


def _table_spec(vibrating: bool = False) -> mujoco.MjSpec:
  """The table.  When ``vibrating``, its top rides on two slide joints, x and z, so a motor can shake it."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table", pos=(TABLE_CENTER_X, 0.0, TABLE_TOP))
  top = body
  if vibrating:
    top = body.add_body(name="top_body", gravcomp=1.0)
    top.add_joint(name="shake_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(1, 0, 0), range=(-0.02, 0.02))
    top.add_joint(name="shake_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(0, 0, 1), range=(-0.02, 0.02))
  # Dream Labs' calibrated YAM station top camera, moved with the arms: it sits where it does relative to the midpoint
  # of the station's arm bases (their 0.2525, 0, 0.76), here our midpoint (0, 0, TABLE_TOP).
  from sim.robot import _station_camera
  _station_camera(body, "station_top", (0.0860 - 0.2525 - TABLE_CENTER_X, -0.0090, 1.7043 - 0.76),
                  (0.68301, 0.18301, -0.18301, -0.68301), (392.1956, 391.7224), (318.3894, 237.8763))
  # Straight down over the arms' working area; a camera looks along its own -z.
  body.add_camera(name="overhead", pos=(0.28 - TABLE_CENTER_X, 0.0, 0.85), fovy=55.0)
  half_x, half_y = TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2
  top.add_geom(
    name="top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(half_x, half_y, TABLE_THICKNESS / 2),
    pos=(0.0, 0.0, -TABLE_THICKNESS / 2),
    mass=TABLE_TOP_MASS,
    rgba=(0.62, 0.48, 0.34, 1.0),
    friction=(0.8, 5e-3, 5e-4),
  )
  leg_half = (TABLE_TOP - TABLE_THICKNESS) / 2
  for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
    body.add_geom(
      name=f"leg_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.025, 0.025, leg_half),
      pos=(sx * (half_x - LEG_INSET), sy * (half_y - LEG_INSET), -TABLE_THICKNESS - leg_half),
      rgba=(0.25, 0.25, 0.27, 1.0),
    )
  return spec


def _cube_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(CUBE_HALF,) * 3,
    mass=CUBE_MASS,
    friction=(1.0, 5e-3, 5e-4),
    rgba=(0.85, 0.2, 0.2, 1.0),
  )
  return spec


def _vibrating_table_cfg() -> EntityCfg:
  """The table with its top on a stiff, critically damped motor, stiff enough to follow 12 Hz."""
  stiffness = 1e6
  return EntityCfg(
    spec_fn=lambda: _table_spec(vibrating=True),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("shake_[xz]",),
          stiffness=stiffness,
          damping=2.0 * math.sqrt(stiffness * TABLE_TOP_MASS),
          effort_limit=20000.0,
        ),
      ),
    ),
  )


def make_rig(ready_joints: dict[str, float], vibrating_table: bool = False) -> dict[str, EntityCfg]:
  """The two arms on the table."""
  entities = {
    arm: get_yam_cfg(pos, rot, ready_joints, *_SHELL_BITS[arm]) for arm, (pos, rot) in BASE_POSE.items()
  }
  entities["table"] = _vibrating_table_cfg() if vibrating_table else EntityCfg(spec_fn=_table_spec)
  return entities


def make_entities(ready_joints: dict[str, float]) -> dict[str, EntityCfg]:
  """The rig and a cube."""
  return {
    **make_rig(ready_joints),
    "cube": EntityCfg(
      spec_fn=_cube_spec,
      init_state=EntityCfg.InitialStateCfg(pos=(0.25, 0.0, TABLE_TOP + CUBE_HALF)),
    ),
  }


def world_to_base(arm: str, point: tuple[float, float, float]) -> tuple[float, float, float]:
  """A point in the world (env) frame, in ``arm``'s base frame, whose z = 0 is the table top."""
  (bx, by, bz), (w, _, _, z) = BASE_POSE[arm]
  yaw = 2.0 * math.atan2(z, w)
  dx, dy = point[0] - bx, point[1] - by
  return (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy, point[2] - bz)
