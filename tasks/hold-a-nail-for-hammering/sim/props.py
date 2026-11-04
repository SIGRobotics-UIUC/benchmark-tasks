"""The objects: a hammer and its rest, a long nail and the rack it starts on, and the post it is driven into.

Each function returns an ``MjSpec`` with one root body.  Props that stay put have no joint; props that can be picked
up have a free joint."""

from __future__ import annotations

import math

import mujoco

BOX = mujoco.mjtGeom.mjGEOM_BOX
CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
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
HAMMER_HANDLE = 0.26
"""Handle length, metres, along the hammer's x; the head sits across the +x end."""
HAMMER_HANDLE_RADIUS = 0.012
HANDLE_FRICTION = (2.0, 0.02, 5e-4)
"""Sliding 2.0, a textured rubber grip; torsional 2 cm, as a pad pressed flat on the handle grips over its area.  MuJoCo
contacts touch at points, and with the grip's default 5 mm the pinch let the hammer turn head down about the jaw
(the pads' normal) in most pick-ups, however light it was.  The pads mix to the larger value, so only the handle
changes."""
HAMMER_HEAD = (0.015, 0.015, 0.035)
"""Half sizes of the steel head: upright, across the handle's +x end, its striking face its underside."""
NARROW_HAMMER_HEAD = (0.010, 0.010, 0.035)
"""A 2 cm square head, for the held-nail tasks: room beside it for the fingers holding the nail."""


def hammer_spec(head: tuple[float, float, float] = HAMMER_HEAD) -> mujoco.MjSpec:
  """A claw-less hammer, origin at the handle's middle.  Put down, it stands on its face with the handle level, the
  handle's far end on a rest.  ``face`` is the site in the middle of the striking face.  ``head``: its half sizes."""
  head_x = HAMMER_HANDLE / 2 + head[0] - 0.005
  spec, body = _root("hammer", free=True)
  # A flat-sided handle, as on many hammers: pinched across, it lies on each pad along a face, which holds the head up.
  # A round one touches each pad at a single point, and the head swings down about the jaw.
  body.add_geom(
    type=BOX, size=(HAMMER_HANDLE / 2, HAMMER_HANDLE_RADIUS * 0.85, HAMMER_HANDLE_RADIUS), mass=0.05,
    rgba=(0.55, 0.36, 0.2, 1.0), friction=HANDLE_FRICTION, condim=4,
  )
  # A light, tack-hammer head, 0.13 kg in all: the pads' 50 N squeeze only just holds a 0.15 kg head out level on the
  # handle, and picked up from the table it tipped in the grip most of the time.
  body.add_geom(type=BOX, size=head, pos=(head_x, 0, 0), mass=0.08, rgba=(0.3, 0.31, 0.33, 1.0), friction=(0.6, 5e-3, 5e-4))
  body.add_site(name="face", pos=(head_x, 0, -head[2]), size=(0.004, 0, 0), rgba=(1, 0, 0, 1))
  for geom in spec.geoms:
    geom.contype = geom.conaffinity = 1
  return _stiffen(spec)
HAMMER_REST = (0.04, 0.08, HAMMER_HEAD[2] - HAMMER_HANDLE_RADIUS)
"""Full sizes of the wooden block the hammer's handle lies on: its top is where the level handle's underside is."""


def hammer_rest_spec() -> mujoco.MjSpec:
  spec, body = _root("hammer_rest", free=False)
  x, y, z = HAMMER_REST
  body.add_geom(type=BOX, size=(x / 2, y / 2, z / 2), pos=(0, 0, z / 2), rgba=(0.6, 0.45, 0.28, 1.0), friction=(0.8, 5e-3, 5e-4))
  return spec
NAIL_FRICTION_SOLIMP = (0.9999, 0.9999, 0.001, 0.5, 2.0)
NAIL_FRICTION_SOLREF = (0.01, 1.0)
SPIKE_SHANK = (0.004, 0.05)
"""Radius and half length of the loose nail's shank, metres: a 10 cm spike, 8 mm across.  Long enough that a hand
holding it low has its fingers 8 cm below the head, out of the hammer's way."""
SPIKE_HEAD = (0.011, 0.0015)
"""Radius and half thickness of its head."""
POST = (0.10, 0.10, 0.06)
"""The wooden post it is driven into: full sizes."""
POST_HOLE = 0.0046
"""Half width of the square hole down the post's middle: 0.6 mm clear of the shank on each side, so a nail set 1 cm in stands within 4 degrees of upright."""
POST_SOCKET = 0.004
"""How deep the nail's point goes into the hole before it is driven: not enough to stand up by itself."""
POST_TRAVEL = 0.05
SPIKE_DRIVE = 0.04
"""How far the held nail must be driven in: 4 cm, firmly set, its head 6 cm proud of the post."""
POST_FRICTION = 20.0
"""Newtons of dry friction holding the plug, the wood's grip on the nail."""


def spike_spec() -> mujoco.MjSpec:
  """A loose nail along its z, head up at +z, origin at the shank's middle.  Sites ``head`` (top of the head) and
  ``tip``."""
  spec, body = _root("spike", free=True)
  steel = (0.7, 0.72, 0.75, 1.0)
  half, length = SPIKE_SHANK
  # Round, with a rounded point: it cannot stand on its point by itself, as a square end would.
  # Heavy for its size (a light nail between stiff contacts made the solver blow up).
  # Twisting friction as the hammer handle has (``HANDLE_FRICTION``): pinched by two pads, it would otherwise pivot
  # between them as soon as the holding hand pressed on it.
  body.add_geom(type=CAPSULE, size=(half, length - half, 0), mass=0.07, rgba=steel, friction=(1.0, 0.02, 5e-4), condim=4)
  radius, thick = SPIKE_HEAD
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(radius, thick, 0), pos=(0, 0, length + thick), mass=0.01,
                rgba=steel, friction=GRIP_FRICTION)
  body.add_site(name="head", pos=(0, 0, length + 2 * thick), size=(0.003, 0, 0), rgba=(1, 0, 0, 1))
  body.add_site(name="tip", pos=(0, 0, -length), size=(0.003, 0, 0), rgba=(1, 0, 0, 1))
  return _stiffen(spec)


def post_spec() -> mujoco.MjSpec:
  """The post, origin at the middle of its underside, with a square hole down its middle.  In the hole rides the plug
  the nail's point stands on: one slide joint, ``plug_drive``, 0 at the start and ``-POST_TRAVEL`` at the bottom, held by
  ``NAIL_FRICTION`` of dry friction as the first hammer task's nails are.  Driving the nail pushes the plug down; the
  plug's depth is how far the nail has gone in."""
  spec, body = _root("post", free=False)
  x, y, z = POST
  h = POST_HOLE
  wood = (0.66, 0.5, 0.32, 1.0)
  # Four slabs around the hole, slippery: the hole is drilled clear, and all the wood's hold on the nail is the plug's.
  # With grip on its walls a nail leaning the least bit in the hole wedges, and blows do not drive it.
  walls = []
  for sign in (1, -1):
    walls.append(body.add_geom(type=BOX, size=(x / 2, (y / 2 - h) / 2, z / 2), pos=(0, sign * (h + (y / 2 - h) / 2), z / 2),
                               rgba=wood, friction=(0.05, 1e-4, 1e-5)))
    walls.append(body.add_geom(type=BOX, size=((x / 2 - h) / 2, h, z / 2), pos=(sign * (h + (x / 2 - h) / 2), 0, z / 2),
                               rgba=wood, friction=(0.05, 1e-4, 1e-5)))
  plug = body.add_body(name="plug", pos=(0, 0, z - POST_SOCKET), gravcomp=1.0)
  plug.add_joint(
    name="plug_drive", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(0, 0, 1), range=(-POST_TRAVEL, 0.0),
    frictionloss=POST_FRICTION, damping=5.0, solimp_friction=NAIL_FRICTION_SOLIMP, solref_friction=NAIL_FRICTION_SOLREF,
  )
  # Its top is at the joint's origin.
  plug.add_geom(type=BOX, size=(h - 0.0005, h - 0.0005, 0.004), pos=(0, 0, -0.004), mass=0.01, rgba=(0.4, 0.3, 0.2, 1.0))
  body.add_site(name="hole", pos=(0, 0, z), size=(0.003, 0, 0), rgba=(1, 0, 0, 1))
  _stiffen(spec)
  for wall in walls:
    wall.priority = 2  # Above the nail's 1, so that the walls' friction, not the nail's grip friction, applies.
  return spec
SPIKE_RACK = (0.03, 0.02, ((0.016, 0.044), (-0.050, -0.034)))
"""The two blocks the nail starts lying across: x size, height, and each block's span along the nail from its middle,
toward the head (+) and toward the point (-).  Between them the shank is clear for fingers to pinch from above where
the holding hand grips it (``SPIKE_GRIP``)."""
SPIKE_GRIP = 0.0
"""How far from the nail's middle, toward the point, the holding hand grips it: low enough that the fingers, held
level, stay well below the head; high enough that the hand can follow the nail down as it is driven."""


def spike_rack_spec(head_sign: float) -> mujoco.MjSpec:
  """``head_sign``: the nail's head points this way along y."""
  spec, body = _root("spike_rack", free=False)
  x, z, spans = SPIKE_RACK
  for lo, hi in spans:
    body.add_geom(type=BOX, size=(x / 2, (hi - lo) / 2, z / 2), pos=(0, head_sign * (lo + hi) / 2, z / 2),
                  rgba=(0.6, 0.45, 0.28, 1.0), friction=(0.8, 5e-3, 5e-4))
  return spec
