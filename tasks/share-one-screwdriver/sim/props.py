"""The objects: a screwdriver and its holder, and two blocks with a screw in each.

Each function returns an ``MjSpec`` with one root body.  Props that stay put have no free joint; the screwdriver has
one, and each screw turns on a hinge in its block."""

from __future__ import annotations

import math

import mujoco

BOX = mujoco.mjtGeom.mjGEOM_BOX
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
HANDLE_FRICTION = (2.0, 0.02, 5e-4)
"""Sliding 2.0, a textured rubber grip; torsional 2 cm, as a pad pressed flat on the handle grips over its area.  MuJoCo
contacts touch at points, and with the grip's default 5 mm the pinch let the hammer turn head down about the jaw
(the pads' normal) in most pick-ups, however light it was.  The pads mix to the larger value, so only the handle
changes."""
PART_FRICTION = (1.0, 0.02, 5e-4)
"""Torsional friction (condim 4) so a part squeezed off-centre does not spin in the fingers."""
DRIVER_HANDLE = (0.015, 0.035)
"""The screwdriver's handle: radius and half length.  The driver's origin is the handle's middle.  3 cm across: the
pads hold a 3 cm part, and let a narrower one slip."""
DRIVER_SHAFT = (0.003, 0.012)
DRIVER_BLADE = (0.008, 0.001, 0.003)
"""Half sizes of each of the two crossed blades at the tip: a cross point."""
DRIVER_TIP = DRIVER_HANDLE[1] + 2 * DRIVER_SHAFT[1] + 2 * DRIVER_BLADE[2]
"""From the driver's origin down to its tip."""
DRIVER_HOLDER = (0.05, 0.05, 0.04)
"""The holder: a block with a hole that the shaft drops into and the handle rests on."""
DRIVER_HOLE = 0.013
"""Half the hole's width: wider than the cross point, narrower than the handle."""
SCREW_BLOCK = (0.06, 0.06, 0.02)
SCREW_SLOT = 0.003
"""Half the width of each arm of the cross slot in the screw's head: the 2 mm blade turns a few degrees in it."""
SCREW_HEAD = (0.012, 0.0015)
"""The head's radius, and half its height above the block."""
SCREW_FRICTION = 0.05
"""N m: what it takes to turn the screw."""
SCREW_TURN = math.pi
"""The screw is home after half a turn."""


def driver_spec() -> mujoco.MjSpec:
  spec, body = _root("driver", free=True)
  hr, hl = DRIVER_HANDLE
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(hr, hl, 0), mass=0.06, rgba=(0.85, 0.2, 0.15, 1.0),
                friction=HANDLE_FRICTION, condim=4)  # A rubber grip, as on the hammer: it must not slide in the pads.
  sr, sl = DRIVER_SHAFT
  steel = (0.7, 0.7, 0.75, 1.0)
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(sr, sl, 0), pos=(0, 0, -hl - sl), mass=0.02, rgba=steel)
  bx, by, bz = DRIVER_BLADE
  for size in ((bx, by, bz), (by, bx, bz)):
    body.add_geom(type=BOX, size=size, pos=(0, 0, -DRIVER_TIP + bz), mass=0.005, rgba=steel, friction=(0.6, 5e-3, 5e-4))
  return _stiffen(spec)


def driver_holder_spec() -> mujoco.MjSpec:
  """A block with a square hole down its middle, open at the top."""
  spec, body = _root("driver_holder", free=False)
  x, y, z = DRIVER_HOLDER
  h = DRIVER_HOLE
  wood = (0.55, 0.4, 0.25, 1.0)
  for sign in (1, -1):
    body.add_geom(type=BOX, size=(x / 2, (y / 2 - h) / 2, z / 2), pos=(0, sign * (h + (y / 2 - h) / 2), z / 2), rgba=wood,
                  friction=(0.1, 1e-3, 1e-4), priority=2)
    body.add_geom(type=BOX, size=((x / 2 - h) / 2, h, z / 2), pos=(sign * (h + (x / 2 - h) / 2), 0, z / 2), rgba=wood,
                  friction=(0.1, 1e-3, 1e-4), priority=2)
  body.add_geom(type=BOX, size=(h, h, 0.005), pos=(0, 0, 0.005), rgba=wood, friction=GRIP_FRICTION)
  return spec


def screw_block_spec(name: str) -> mujoco.MjSpec:
  """A wooden block with a screw in its top: the screw's head turns on a hinge, with friction, and has a cross slot."""
  spec, body = _root(name, free=False)
  x, y, z = SCREW_BLOCK
  body.add_geom(type=BOX, size=(x / 2, y / 2, z / 2), pos=(0, 0, z / 2), rgba=(0.55, 0.4, 0.25, 1.0), friction=GRIP_FRICTION)
  head = body.add_body(name=f"{name}_screw", pos=(0, 0, z))
  head.add_joint(name=f"{name}_turn", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1), frictionloss=SCREW_FRICTION,
                 damping=0.05, armature=1e-3)  # Not lighter: a light head left alone drifts and blows up.
  r, hh = SCREW_HEAD
  steel = (0.75, 0.75, 0.78, 1.0)
  head.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(r, hh * 0.5, 0), pos=(0, 0, hh * 0.5), rgba=steel,
                contype=0, conaffinity=0, mass=0.0)
  # Four quarters of the head around a cross slot: what the blades push on.
  q = (r - SCREW_SLOT) / 2
  for sx in (1, -1):
    for sy in (1, -1):
      head.add_geom(type=BOX, size=(q, q, hh), pos=(sx * (SCREW_SLOT + q), sy * (SCREW_SLOT + q), hh), mass=0.01,
                    rgba=steel, friction=(0.6, 5e-3, 5e-4))
  return spec
