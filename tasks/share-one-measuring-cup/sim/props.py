"""The objects: a measuring cup with sloping sides and two tabs, two bowls, and the beads.

Each function returns an ``MjSpec`` with one root body.  Props that stay put have no free joint; the cup and the
beads have one.  MuJoCo has no granular flow: the beads are low-friction balls."""

from __future__ import annotations

import math

import mujoco

BOX = mujoco.mjtGeom.mjGEOM_BOX
SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
GRIP_FRICTION = (1.0, 5e-3, 5e-4)
OBJECT_SOLREF = (0.01, 1.0)
OBJECT_SOLIMP = (0.99, 0.995, 0.001, 0.5, 2.0)
"""Contact stiffness for everything that can be picked up, as for the finger pads (``sim.robot``).  MuJoCo's default
contact is soft in proportion to how light the object is: with it a 4 mm utensil sank 6.6 mm into the table, the bin and
the trays, out of sight, and a fruit 7 mm into its tray.  Priority 1 makes these settings win against the boxes and the
table; the pads have the same, so a grip is unchanged."""


def _stiffen(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  for geom in spec.geoms:
    if geom.contype or geom.conaffinity:
      geom.solref, geom.solimp, geom.priority = OBJECT_SOLREF, OBJECT_SOLIMP, 1
  return spec


def _root(name: str, free: bool) -> tuple[mujoco.MjSpec, mujoco.MjsBody]:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=name)
  if free:
    body.add_freejoint(name=f"{name}_joint")
  return spec, body


def _open_box(body, inner: tuple[float, float], height: float, wall: float, floor: float, rgba, mass: float = 0.0) -> None:
  """Walls and a floor around the body's origin, which is at the middle of the underside."""
  ix, iy = inner[0] / 2, inner[1] / 2
  density = {} if mass == 0.0 else {"mass": mass / 5}
  body.add_geom(type=BOX, size=(ix + wall, iy + wall, floor / 2), pos=(0, 0, floor / 2), rgba=rgba, friction=GRIP_FRICTION, **density)
  mid = floor + height / 2
  for sign in (1, -1):
    body.add_geom(
      type=BOX, size=(ix + wall, wall / 2, height / 2), pos=(0, sign * (iy + wall / 2), mid), rgba=rgba,
      friction=GRIP_FRICTION, **density,
    )
    body.add_geom(
      type=BOX, size=(wall / 2, iy, height / 2), pos=(sign * (ix + wall / 2), 0, mid), rgba=rgba,
      friction=GRIP_FRICTION, **density,
    )
MCUP_FLOOR = 0.04
"""The measuring cup's square floor, a side: the 16 beads lie on it in one layer."""
MCUP_WALL = (0.028, 40.0, 0.003)
"""Each wall's slant height, how far it leans out from upright (degrees), and its thickness.  Leaning walls pour
before the cup is on its side: a bead goes over the lip once the cup tilts past 90 degrees less the lean."""
MCUP_TAB = (0.03, 0.03, 0.003)
"""The two tabs it is held by, standing up from the tops of two opposite walls: width, height, thickness."""


def mcup_spec() -> mujoco.MjSpec:
  """A measuring cup with leaning walls, and two upright tabs on two of their tops for a gripper to pinch."""
  spec, body = _root("cup", free=True)
  rgba = (0.2, 0.55, 0.65, 1.0)
  half, (slant, lean, thick) = MCUP_FLOOR / 2, MCUP_WALL
  a = math.radians(lean)
  body.add_geom(type=BOX, size=(half + thick, half + thick, 0.002), pos=(0, 0, 0.002), mass=0.03, rgba=rgba,
                friction=GRIP_FRICTION)
  rim = half + slant * math.sin(a)
  top = 0.004 + slant * math.cos(a)
  length = rim + thick
  for axis in (0, 1):
    for sign in (1, -1):
      centre = [0.0, 0.0, 0.004 + slant / 2 * math.cos(a)]
      centre[axis] = sign * (half + slant / 2 * math.sin(a))
      # Turned about the edge it stands on, top out.
      angle = sign * lean if axis == 0 else -sign * lean
      euler = (0.0, angle, 0.0) if axis == 0 else (angle, 0.0, 0.0)
      size = (thick / 2, length, slant / 2) if axis == 0 else (length, thick / 2, slant / 2)
      body.add_geom(type=BOX, size=size, pos=centre, euler=euler, mass=0.01, rgba=rgba, friction=GRIP_FRICTION)
  tw, th, tt = MCUP_TAB
  for sign in (1, -1):
    body.add_geom(type=BOX, size=(tt / 2, tw / 2, th / 2), pos=(sign * (rim + tt / 2), 0, top + th / 2), mass=0.005,
                  rgba=(0.15, 0.4, 0.5, 1.0), friction=GRIP_FRICTION)
  return _stiffen(spec)


def mcup_tab_height() -> float:
  """The middle of the tabs, over the cup's foot: where it is held."""
  return 0.004 + MCUP_WALL[0] * math.cos(math.radians(MCUP_WALL[1])) + MCUP_TAB[1] / 2


def bowl_spec(inner: float, height: float, name: str = "bowl", free: bool = True) -> mujoco.MjSpec:
  spec, body = _root(name, free=free)
  _open_box(body, (inner, inner), height, 0.005, 0.005, rgba=(0.85, 0.82, 0.72, 1.0), mass=0.25)
  return _stiffen(spec)
BEAD_RADIUS = 0.006


def bead_spec() -> mujoco.MjSpec:
  """One drop of the liquid.  MuJoCo has no fluid, so the pour is a cup of low-friction beads."""
  spec, body = _root("bead", free=True)
  body.add_geom(type=SPHERE, size=(BEAD_RADIUS, 0, 0), mass=0.002, rgba=(0.2, 0.45, 0.9, 1.0), friction=(0.1, 1e-3, 1e-4))
  return _stiffen(spec)
