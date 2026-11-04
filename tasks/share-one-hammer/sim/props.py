"""The objects: a hammer and its rest, and two wooden blocks with a nail standing in each.

Each function returns an ``MjSpec`` with one root body.  Props that stay put have no free joint; the hammer has one."""

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
HAMMER_HEAD_X = HAMMER_HANDLE / 2 + HAMMER_HEAD[0] - 0.005
"""The head's centre along the handle, from the handle's middle."""


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
    geom.contype = geom.conaffinity = 1 | NAIL_BIT
  return _stiffen(spec)
NAIL_BIT = 8
"""The contact bit the nail heads and the hammer share, and nothing else: only the hammer can drive a nail, not a finger
or an arm pressing on it.  (The arms' shells are on bits 2 and 4, everything else on 1.)"""
HAMMER_REST = (0.04, 0.08, HAMMER_HEAD[2] - HAMMER_HANDLE_RADIUS)
"""Full sizes of the wooden block the hammer's handle lies on: its top is where the level handle's underside is."""


def hammer_rest_spec() -> mujoco.MjSpec:
  spec, body = _root("hammer_rest", free=False)
  x, y, z = HAMMER_REST
  body.add_geom(type=BOX, size=(x / 2, y / 2, z / 2), pos=(0, 0, z / 2), rgba=(0.6, 0.45, 0.28, 1.0), friction=(0.8, 5e-3, 5e-4))
  return spec
NAIL_BLOCK = (0.08, 0.08, 0.03)
"""The wooden block the nail goes into: full sizes, metres.  Low, like the hammer: with its fingers pointing down the
arm's grasp site reaches only about 0.2 m above the table (joint 4's limits), so a tall stack would leave no room to
lift the hammer over the nail."""
NAIL_STANDOUT = 0.025
"""How far the nail's head stands above the block at the start; driving it home takes it all the way down."""
NAIL_HEAD_RADIUS = 0.01
NAIL_FRICTION = 50.0
"""Newtons of dry friction holding the nail in the wood (``python -m sim.check_hammer``).  The hammer lying on it does
not move it, and dropped on it from 5 cm drives it under 2 mm; brought down on it in the gripper, ten blows of a
scripted arm drive it home.  A firm push with the hammer drives it too: the rule is that the hammer does the work, and
only the hammer touches the nail heads (``NAIL_BIT``).  MuJoCo's dry friction is a soft constraint that yields slowly to
any steady force; the stiff impedance below keeps that creep to nothing under a resting hammer."""
NAIL_FRICTION_SOLIMP = (0.9999, 0.9999, 0.001, 0.5, 2.0)
NAIL_FRICTION_SOLREF = (0.01, 1.0)


def nail_block_spec(name: str) -> mujoco.MjSpec:
  """A block of wood with a nail standing in it, origin at the middle of the block's underside.  The nail rides on one
  slide joint, ``<name>_drive``, at 0 at the start and ``-NAIL_STANDOUT`` driven home; ``head`` is the site on its head."""
  spec, body = _root(name, free=False)
  bx, by, bz = NAIL_BLOCK
  body.add_geom(type=BOX, size=(bx / 2, by / 2, bz / 2), pos=(0, 0, bz / 2), rgba=(0.78, 0.62, 0.4, 1.0), friction=(0.8, 5e-3, 5e-4))
  # Weightless, so that only what hits it moves it.
  nail = body.add_body(name=f"{name}_nail", pos=(0, 0, bz + NAIL_STANDOUT), gravcomp=1.0)
  nail.add_joint(
    name=f"{name}_drive", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(0, 0, 1), range=(-NAIL_STANDOUT, 0.0),
    frictionloss=NAIL_FRICTION, damping=5.0, solimp_friction=NAIL_FRICTION_SOLIMP, solref_friction=NAIL_FRICTION_SOLREF,
  )
  steel = (0.7, 0.72, 0.75, 1.0)
  nail.add_geom(
    type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(NAIL_HEAD_RADIUS, 0.0015, 0), pos=(0, 0, -0.0015), mass=0.01, rgba=steel,
    contype=0, conaffinity=0,
  )
  # What the hammer hits: a block under the head, deep into the wood (only the hammer touches it).  A blow moves the
  # hammer 5 mm in a 5 ms step, through the thin head itself, which would then be pushed out upward.
  nail.add_geom(
    type=BOX, size=(NAIL_HEAD_RADIUS * 0.85, NAIL_HEAD_RADIUS * 0.85, 0.012), pos=(0, 0, -0.012), mass=0.0,
    contype=NAIL_BIT, conaffinity=NAIL_BIT, group=3, rgba=(1, 0, 0, 0.3),
  )
  # The shank is only seen: inside the wood it would collide with the block.
  nail.add_geom(
    type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(0.0025, 0.025, 0), pos=(0, 0, -0.026), mass=0.01, rgba=steel,
    contype=0, conaffinity=0,
  )
  nail.add_site(name="head", pos=(0, 0, 0), size=(0.003, 0, 0), rgba=(1, 0, 0, 1))
  return _stiffen(spec)
