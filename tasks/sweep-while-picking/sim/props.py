"""The objects: a hand broom and the board it rests on, Arm 1's bin, the red blocks Arm 1 picks, and the clutter the
controlled arm sweeps off the table.

Each function returns an ``MjSpec`` with one root body.  Props that stay put have no joint; props that can be moved have
a free joint."""

from __future__ import annotations

import math

import mujoco

BOX = mujoco.mjtGeom.mjGEOM_BOX
SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
OBJECT_SOLREF = (0.01, 1.0)
OBJECT_SOLIMP = (0.99, 0.995, 0.001, 0.5, 2.0)
"""Contact stiffness for everything that can move, as for the finger pads (``sim.robot``): MuJoCo's default contact is
soft in proportion to how light the object is, and light things sink into the table."""
WOOD = (0.66, 0.5, 0.32, 1.0)


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


# --- The hand broom -----------------------------------------------------------------------------------------------
BRUSH = (0.20, 0.05, 0.02)
"""Full sizes of the broom's plastic back: 20 cm long (its x), 5 cm wide, 2 cm thick."""
BRISTLES = (0.20, 0.04, 0.025)
"""Full sizes of the bristle block under it, modelled as one solid, slightly soft block."""
HANDLE = (0.08, 0.020, 0.030)
"""The grip on top: 8 cm long, 2 cm across (the jaw closes across it), 3 cm deep, so the pads close on most of it."""
HANDLE_TILT = 30.0
"""Degrees the grip rises toward the broom's front (+x), as a hand broom's handle is angled up off its back: held with the
fingers leaning forward by as much, which is how this arm reaches out over the table, the bristles lie flat."""
HANDLE_FRICTION = (2.0, 0.05, 5e-4)
"""A textured rubber grip; torsional 5 cm, as two flat pads pressed on a flat-sided grip resist its turning between
them (with less, the broom swung down about the jaw when its bristles caught)."""
GRIP_HEIGHT = 0.092
"""The middle of the grip's axis above the bristles' tips: where the hand holds it."""


def broom_spec() -> mujoco.MjSpec:
  """A hand broom, origin at the bristles' tips under its middle, length along x, its grip on two posts over its back,
  rising toward +x.  Site ``grip``: the middle of the grip, where the hand holds it."""
  spec, body = _root("broom", free=True)
  plastic, bristle = (0.2, 0.35, 0.7, 1.0), (0.15, 0.15, 0.15, 1.0)
  body.add_geom(type=BOX, size=(BRISTLES[0] / 2, BRISTLES[1] / 2, BRISTLES[2] / 2), pos=(0, 0, BRISTLES[2] / 2),
                mass=0.02, rgba=bristle, friction=(0.5, 5e-3, 5e-4))
  back = BRISTLES[2] + BRUSH[2]
  body.add_geom(type=BOX, size=(BRUSH[0] / 2, BRUSH[1] / 2, BRUSH[2] / 2), pos=(0, 0, back - BRUSH[2] / 2), mass=0.03,
                rgba=plastic, friction=(0.6, 5e-3, 5e-4))
  t = math.radians(HANDLE_TILT)
  axis = (math.cos(t), 0.0, math.sin(t))
  under = HANDLE[2] / 2 / math.cos(t)  # From the grip's axis down to its underside, vertically.
  for end in (-0.036, 0.036):  # A post under each end of the grip, clear of the fingers.
    x, top = end * axis[0], GRIP_HEIGHT + end * axis[2] - under + 0.003
    body.add_geom(type=BOX, size=(0.006, 0.009, (top - back) / 2), pos=(x, 0, (top + back) / 2), mass=0.005,
                  rgba=plastic)
  half = math.radians(-HANDLE_TILT) / 2
  body.add_geom(type=BOX, size=(HANDLE[0] / 2, HANDLE[1] / 2, HANDLE[2] / 2), pos=(0, 0, GRIP_HEIGHT),
                quat=(math.cos(half), 0, math.sin(half), 0), mass=0.02, rgba=(0.2, 0.2, 0.22, 1.0),
                friction=HANDLE_FRICTION, condim=4)
  body.add_site(name="grip", pos=(0, 0, GRIP_HEIGHT), size=(0.004, 0, 0), rgba=(1, 0, 0, 1))
  for geom in spec.geoms:
    geom.contype = geom.conaffinity = 1
  return _stiffen(spec)


BROOM_REST = (0.24, 0.07, 0.015)
"""Full sizes of the wooden board the broom starts standing on."""


def broom_rest_spec() -> mujoco.MjSpec:
  spec, body = _root("broom_rest", free=False)
  x, y, z = BROOM_REST
  body.add_geom(type=BOX, size=(x / 2, y / 2, z / 2), pos=(0, 0, z / 2), rgba=WOOD, friction=(0.8, 5e-3, 5e-4))
  return spec


# --- Arm 1's bin and blocks ----------------------------------------------------------------------------------------
BIN = (0.15, 0.15, 0.06, 0.01)
"""Inside length, inside width, wall height and wall thickness of Arm 1's grey plastic bin."""


def bin_spec() -> mujoco.MjSpec:
  """Origin at the middle of its floor's underside; fixed to the table."""
  spec, body = _root("bin", free=False)
  ix, iy, h, t = BIN
  grey = (0.55, 0.56, 0.58, 1.0)
  body.add_geom(type=BOX, size=(ix / 2 + t, iy / 2 + t, t / 2), pos=(0, 0, t / 2), rgba=grey)
  for s in (1, -1):
    body.add_geom(type=BOX, size=(t / 2, iy / 2 + t, h / 2), pos=(s * (ix / 2 + t / 2), 0, h / 2), rgba=grey)
    body.add_geom(type=BOX, size=(ix / 2, t / 2, h / 2), pos=(0, s * (iy / 2 + t / 2), h / 2), rgba=grey)
  return spec


BLOCK = 0.04
"""Arm 1's red blocks: 4 cm wooden cubes, 40 g."""


def red_block_spec() -> mujoco.MjSpec:
  spec, body = _root("red_block", free=True)
  body.add_geom(type=BOX, size=(BLOCK / 2,) * 3, mass=0.04, rgba=(0.85, 0.1, 0.1, 1.0), friction=(0.8, 5e-3, 5e-4),
                condim=4)
  for geom in spec.geoms:
    geom.contype = geom.conaffinity = 1
  return _stiffen(spec)


# --- Clutter -------------------------------------------------------------------------------------------------------
CLUTTER_KINDS = {
  # kind: (description, builder args)
  "cube": "painted wooden cube, 3 cm, 20 g",
  "ball": "rubber ball, 4 cm diameter, 30 g",
  "box": "empty plastic box, 6 x 4 x 3 cm, lid closed, 20 g",
  "paper": "crumpled paper ball, about 4.5 cm across, 10 g",
}
CLUTTER_COLORS = {"cube": (0.2, 0.45, 0.85, 1.0), "ball": (0.95, 0.8, 0.15, 1.0), "box": (0.2, 0.7, 0.35, 1.0),
                  "paper": (0.93, 0.93, 0.9, 1.0)}


def clutter_half_height(kind: str) -> float:
  return {"cube": 0.015, "ball": 0.02, "box": 0.015, "paper": 0.0225}[kind]


def clutter_spec(kind: str) -> mujoco.MjSpec:
  spec, body = _root("clutter", free=True)
  rgba = CLUTTER_COLORS[kind]
  if kind == "cube":
    body.add_geom(type=BOX, size=(0.015,) * 3, mass=0.02, rgba=rgba, friction=(0.7, 5e-3, 5e-4))
  elif kind == "ball":
    body.add_geom(type=SPHERE, size=(0.02, 0, 0), mass=0.03, rgba=rgba, friction=(0.9, 5e-3, 2e-3), condim=6)
  elif kind == "box":
    body.add_geom(type=BOX, size=(0.03, 0.02, 0.015), mass=0.02, rgba=rgba, friction=(0.5, 5e-3, 5e-4))
  elif kind == "paper":
    # Crumpled paper hardly rolls: high rolling friction.  A sphere, not a lumpy shape: cylinders and ellipsoids
    # collide through MuJoCo's general convex test, one contact at a time, and a light one spun up until the solver
    # diverged.
    body.add_geom(type=SPHERE, size=(0.0225, 0, 0), mass=0.01, rgba=rgba, friction=(0.8, 0.01, 0.02), condim=6)
  else:
    raise ValueError(kind)
  for geom in spec.geoms:
    geom.contype = geom.conaffinity = 1
  return _stiffen(spec)
