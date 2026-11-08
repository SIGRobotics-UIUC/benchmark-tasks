"""The rig: two YAM Pro arms side by side on one edge of a table, both reaching across it.

    y
    ^   +-----------------------+
    |   |[L] ->                 |       Seen from above.  The origin is midway between the arm bases, on the floor;
    0 --|                       |--> x  x forward, y toward the left arm.  ``TABLE_TOP`` is the table's top.
    |   |[R] ->                 |
        +-----------------------+

* Base separation, 0.50 m.
* 0.90 m front to back; 0.90 m across, plus 0.30 m more on the left arm's side past a painted
  white line, with a 3 cm lip around that part's three open sides.  The bases sit flush with the back edge: their back faces, 35 mm behind the base axes, are on it.
* Height, 0.75 m, over a floor plane: things that go over the edge fall and stay fallen.
"""

from __future__ import annotations

import math

import mujoco

from mjlab.entity import EntityCfg

from sim.robot import _station_camera, get_yam_cfg

TABLE_TOP = 0.75
TABLE_SIZE = (0.90, 0.90)
"""x (front to back) by y (side to side), metres."""
EXTENSION = 0.30
"""The table runs on this far past ``TABLE_SIZE`` on the controlled (left, +y) arm's side, so swept items land on the
table, not the floor."""
LINE_Y = TABLE_SIZE[1] / 2
"""A white line is painted across the table here, where the square part ends; items swept past it count as cleared."""
LINE_WIDTH = 0.02
LIP = (0.01, 0.03)
"""A wooden lip, 1 cm thick and 3 cm high, around the extension's three open sides, so nothing swept or rolled past the
line goes over its edges."""
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


def _table_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table", pos=(TABLE_CENTER_X, 0.0, TABLE_TOP))
  # Dream Labs' calibrated YAM station top camera, moved with the arms: it sits where it does relative to the midpoint
  # of the station's arm bases (their 0.2525, 0, 0.76), here our midpoint (0, 0, TABLE_TOP).
  _station_camera(body, "station_top", (0.0860 - 0.2525 - TABLE_CENTER_X, -0.0090, 1.7043 - 0.76),
                  (0.68301, 0.18301, -0.18301, -0.68301), (392.1956, 391.7224), (318.3894, 237.8763))
  # Straight down over the arms' working area; a camera looks along its own -z.
  body.add_camera(name="overhead", pos=(0.28 - TABLE_CENTER_X, 0.0, 0.85), fovy=55.0)
  half_x, half_y = TABLE_SIZE[0] / 2, (TABLE_SIZE[1] + EXTENSION) / 2
  mid_y = EXTENSION / 2  # The top runs from -TABLE_SIZE[1] / 2 to LINE_Y + EXTENSION.
  body.add_geom(
    name="top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(half_x, half_y, TABLE_THICKNESS / 2),
    pos=(0.0, mid_y, -TABLE_THICKNESS / 2),
    mass=TABLE_TOP_MASS,
    rgba=(0.62, 0.48, 0.34, 1.0),
    friction=(0.8, 5e-3, 5e-4),
  )
  # The painted line: visual only, flush with the top.
  body.add_geom(name="line", type=mujoco.mjtGeom.mjGEOM_BOX, size=(half_x, LINE_WIDTH / 2, 0.0005),
                pos=(0.0, LINE_Y, 0.0005), rgba=(0.95, 0.95, 0.95, 1.0), contype=0, conaffinity=0)
  far = LINE_Y + EXTENSION
  lip_t, lip_h = LIP
  for name, size, at in (
    ("lip_end", (half_x, lip_t / 2, lip_h / 2), (0.0, far - lip_t / 2, lip_h / 2)),
    ("lip_front", (lip_t / 2, EXTENSION / 2, lip_h / 2), (half_x - lip_t / 2, LINE_Y + EXTENSION / 2, lip_h / 2)),
    ("lip_back", (lip_t / 2, EXTENSION / 2, lip_h / 2), (-half_x + lip_t / 2, LINE_Y + EXTENSION / 2, lip_h / 2)),
  ):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=at, rgba=(0.55, 0.42, 0.30, 1.0),
                  friction=(0.8, 5e-3, 5e-4))
  leg_half = (TABLE_TOP - TABLE_THICKNESS) / 2
  for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
    body.add_geom(
      name=f"leg_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.025, 0.025, leg_half),
      pos=(sx * (half_x - LEG_INSET), mid_y + sy * (half_y - LEG_INSET), -TABLE_THICKNESS - leg_half),
      rgba=(0.25, 0.25, 0.27, 1.0),
    )
  return spec


def make_rig(ready_joints: dict[str, float]) -> dict[str, EntityCfg]:
  """The two arms on the table."""
  entities = {
    arm: get_yam_cfg(pos, rot, ready_joints, *_SHELL_BITS[arm]) for arm, (pos, rot) in BASE_POSE.items()
  }
  entities["table"] = EntityCfg(spec_fn=_table_spec)
  return entities


def world_to_base(arm: str, point: tuple[float, float, float]) -> tuple[float, float, float]:
  """A point in the world (env) frame, in ``arm``'s base frame, whose z = 0 is the table top."""
  (bx, by, bz), (w, _, _, z) = BASE_POSE[arm]
  yaw = 2.0 * math.atan2(z, w)
  dx, dy = point[0] - bx, point[1] - by
  return (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy, point[2] - bz)
