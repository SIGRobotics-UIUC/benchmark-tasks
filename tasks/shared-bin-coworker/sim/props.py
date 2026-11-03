"""The objects the benchmark scenes are built from.  Each function returns an ``MjSpec`` with one root body.

Props that stay put have no joint, and mjlab places them per env; props that can be picked up have a free joint.
Every collision geom is on contact bit 1, with the table, the fingertip pads and the arm shells.
"""

from __future__ import annotations

import math

import mujoco

BOX = mujoco.mjtGeom.mjGEOM_BOX
SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
GRIP_FRICTION = (1.0, 5e-3, 5e-4)
FRUIT_CONTACT = {"condim": 6, "friction": (1.0, 5e-3, 2e-3)}
"""With rolling friction: a sphere without it never comes to rest."""


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


BIN_FLOOR = 0.02
"""Thick enough that a flat utensil cannot tunnel through it in a 5 ms step."""


def bin_spec(inner: tuple[float, float], height: float) -> mujoco.MjSpec:
  spec, body = _root("bin", free=False)
  _open_box(body, inner, height, wall=0.008, floor=BIN_FLOOR, rgba=(0.32, 0.38, 0.45, 1.0))
  return spec


def tray_spec(name: str, inner: tuple[float, float], rgba) -> mujoco.MjSpec:
  spec, body = _root(name, free=False)
  _open_box(body, inner, 0.012, wall=0.006, floor=0.004, rgba=rgba)
  return spec


def fruit_spec(kind: str) -> mujoco.MjSpec:
  spec, body = _root(kind, free=True)
  if kind == "apple":
    body.add_geom(type=SPHERE, size=(0.035, 0, 0), mass=0.15, rgba=(0.75, 0.1, 0.12, 1.0), **FRUIT_CONTACT)
  elif kind == "orange":
    body.add_geom(type=SPHERE, size=(0.033, 0, 0), mass=0.13, rgba=(0.95, 0.55, 0.1, 1.0), **FRUIT_CONTACT)
  elif kind == "banana":
    # Straight: a bent pair of capsules rocks for ever on the bin floor.
    body.add_geom(type=CAPSULE, size=(0.018, 0.06, 0), euler=(90, 0, 0), mass=0.12, rgba=(0.95, 0.85, 0.25, 1.0), **FRUIT_CONTACT)
  else:
    raise ValueError(kind)
  return _stiffen(spec)


def utensil_spec(kind: str) -> mujoco.MjSpec:
  """A flat handle along y with a head on the +y end, 16 cm in all."""
  spec, body = _root(kind, free=True)
  steel = (0.75, 0.77, 0.8, 1.0)
  body.add_geom(type=BOX, size=(0.008, 0.05, 0.004), pos=(0, -0.03, 0), mass=0.03, rgba=steel, friction=GRIP_FRICTION)
  if kind == "fork":
    for x in (-0.009, 0.0, 0.009):
      body.add_geom(type=BOX, size=(0.003, 0.03, 0.003), pos=(x, 0.05, 0), mass=0.005, rgba=steel, friction=GRIP_FRICTION)
    body.add_geom(type=BOX, size=(0.012, 0.004, 0.003), pos=(0, 0.022, 0), mass=0.002, rgba=steel, friction=GRIP_FRICTION)
  elif kind == "spoon":
    body.add_geom(type=ELLIPSOID, size=(0.02, 0.03, 0.006), pos=(0, 0.05, 0), mass=0.015, rgba=steel, friction=GRIP_FRICTION)
  elif kind == "knife":
    body.add_geom(type=BOX, size=(0.012, 0.03, 0.002), pos=(0, 0.05, 0), mass=0.015, rgba=steel, friction=GRIP_FRICTION)
  else:
    raise ValueError(kind)
  return _stiffen(spec)


MARKER_LENGTH = 0.13
"""Tip to cap, metres."""


def marker_spec() -> mujoco.MjSpec:
  """A marker along z, origin at its middle, tip down.  ``tip`` is the site at the writing end."""
  spec, body = _root("marker", free=True)
  radius = 0.011
  body.add_geom(
    type=CAPSULE, size=(radius, MARKER_LENGTH / 2 - radius, 0), mass=0.02, rgba=(0.12, 0.25, 0.75, 1.0),
    friction=GRIP_FRICTION,
  )
  body.add_site(name="tip", pos=(0, 0, -MARKER_LENGTH / 2), size=(0.002, 0, 0), rgba=(0, 0, 0, 1))
  return _stiffen(spec)


def stand_spec(inner: float, height: float) -> mujoco.MjSpec:
  """A square tube that holds the marker upright where both arms can reach it."""
  spec, body = _root("stand", free=False)
  _open_box(body, (inner, inner), height, wall=0.006, floor=0.004, rgba=(0.3, 0.3, 0.33, 1.0))
  return spec


def _outline(body, points: list[tuple[float, float]], z: float) -> None:
  """A thin, contactless line through ``points``, closed, for showing a drawing's target."""
  for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
    length = math.hypot(x1 - x0, y1 - y0)
    yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))
    body.add_geom(
      type=BOX, size=(length / 2 + 0.001, 0.0025, 0.0003), pos=((x0 + x1) / 2, (y0 + y1) / 2, z), euler=(0, 0, yaw),
      contype=0, conaffinity=0, mass=0.0, rgba=(0.75, 0.75, 0.78, 1.0),
    )


BOARD_THICKNESS = 0.01


def board_spec(name: str, size: tuple[float, float], outline: list[tuple[float, float]]) -> mujoco.MjSpec:
  """A flat whiteboard, origin at the middle of its underside."""
  spec, body = _root(name, free=False)
  body.add_geom(
    type=BOX, size=(size[0] / 2, size[1] / 2, BOARD_THICKNESS / 2), pos=(0, 0, BOARD_THICKNESS / 2),
    rgba=(0.96, 0.96, 0.94, 1.0), friction=(0.4, 5e-3, 5e-4),
  )
  _outline(body, outline, BOARD_THICKNESS + 0.0003)
  return spec


CUP_INNER = 0.054
CUP_HEIGHT = 0.08
CUP_WALL = 0.004
CUP_FLOOR = 0.004


def cup_spec() -> mujoco.MjSpec:
  spec, body = _root("cup", free=True)
  _open_box(body, (CUP_INNER, CUP_INNER), CUP_HEIGHT, CUP_WALL, CUP_FLOOR, rgba=(0.2, 0.55, 0.65, 1.0), mass=0.08)
  return _stiffen(spec)


def bowl_spec(inner: float, height: float) -> mujoco.MjSpec:
  spec, body = _root("bowl", free=True)
  _open_box(body, (inner, inner), height, 0.005, 0.005, rgba=(0.85, 0.82, 0.72, 1.0), mass=0.25)
  return _stiffen(spec)


BEAD_RADIUS = 0.006


def bead_spec() -> mujoco.MjSpec:
  """One drop of the liquid.  MuJoCo has no fluid, so the pour is a cup of low-friction beads."""
  spec, body = _root("bead", free=True)
  body.add_geom(type=SPHERE, size=(BEAD_RADIUS, 0, 0), mass=0.002, rgba=(0.2, 0.45, 0.9, 1.0), friction=(0.1, 1e-3, 1e-4))
  return _stiffen(spec)


SANDER_FOOT = (0.08, 0.06)


def sander_spec() -> mujoco.MjSpec:
  """A power sander: a flat foot and a handle to pinch across y.  Origin at the middle of the underside."""
  spec, body = _root("sander", free=True)
  body.add_geom(
    type=BOX, size=(SANDER_FOOT[0] / 2, SANDER_FOOT[1] / 2, 0.015), pos=(0, 0, 0.015), mass=0.3,
    rgba=(0.85, 0.4, 0.1, 1.0), friction=(0.6, 5e-3, 5e-4),
  )
  body.add_geom(
    type=BOX, size=(0.03, 0.0125, 0.03), pos=(0, 0, 0.06), mass=0.05, rgba=(0.15, 0.15, 0.17, 1.0),
    friction=GRIP_FRICTION,
  )
  return _stiffen(spec)


def patch_spec(size: tuple[float, float]) -> mujoco.MjSpec:
  """The rough board to sand, origin at the middle of its underside."""
  spec, body = _root("patch", free=False)
  body.add_geom(
    type=BOX, size=(size[0] / 2, size[1] / 2, 0.002), pos=(0, 0, 0.002), rgba=(0.45, 0.3, 0.18, 1.0),
    friction=(0.6, 5e-3, 5e-4),
  )
  return spec
