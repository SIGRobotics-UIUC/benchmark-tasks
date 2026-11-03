"""The objects on the table.  Each function returns an ``MjSpec`` with one root body.

Props that stay put have no joint, and mjlab places them per env; props that can be picked up have a free joint.
Every collision geom is on contact bit 1, with the table, the fingertip pads and the arm shells.
"""

from __future__ import annotations

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


def _open_box(body, inner: tuple[float, float], height: float, wall: float, floor: float, rgba) -> None:
  """Walls and a floor around the body's origin, which is at the middle of the underside."""
  ix, iy = inner[0] / 2, inner[1] / 2
  body.add_geom(type=BOX, size=(ix + wall, iy + wall, floor / 2), pos=(0, 0, floor / 2), rgba=rgba, friction=GRIP_FRICTION)
  mid = floor + height / 2
  for sign in (1, -1):
    body.add_geom(
      type=BOX, size=(ix + wall, wall / 2, height / 2), pos=(0, sign * (iy + wall / 2), mid), rgba=rgba,
      friction=GRIP_FRICTION,
    )
    body.add_geom(
      type=BOX, size=(wall / 2, iy, height / 2), pos=(sign * (ix + wall / 2), 0, mid), rgba=rgba,
      friction=GRIP_FRICTION,
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
