"""The objects: a sponge and two plates with dirt drawn on them.

Each function returns an ``MjSpec`` with one root body.  Props that stay put have no free joint; the sponge has one."""

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
PART_FRICTION = (1.0, 0.02, 5e-4)
"""Torsional friction (condim 4) so a part squeezed off-centre does not spin in the fingers."""
SPONGE = (0.045, 0.07, 0.03)
"""The sponge: width (x), length (y) and height.  Held across its width."""
PLATE_RADIUS = 0.08
PLATE_HEIGHT = 0.01
DIRT = ((0.0, 0.0), (0.035, 0.035), (0.035, -0.035), (-0.035, 0.035), (-0.035, -0.035))
"""Dirty spots on each plate, from its middle: wiped when the sponge passes over them pressed on the plate."""
DIRT_RADIUS = 0.012


def sponge_spec() -> mujoco.MjSpec:
  spec, body = _root("sponge", free=True)
  body.add_geom(type=BOX, size=(SPONGE[0] / 2, SPONGE[1] / 2, SPONGE[2] / 2), mass=0.06, rgba=(0.95, 0.85, 0.2, 1.0),
                friction=PART_FRICTION, condim=4)
  body.add_geom(type=BOX, size=(SPONGE[0] / 2, SPONGE[1] / 2, 0.003), pos=(0, 0, -SPONGE[2] / 2 + 0.003),
                rgba=(0.2, 0.55, 0.3, 1.0), contype=0, conaffinity=0, mass=0.0)  # The green scouring side, underneath.
  return _stiffen(spec)


def plate_spec(name: str) -> mujoco.MjSpec:
  """A flat white plate, fixed to the table, with brown spots of dirt drawn on it."""
  spec, body = _root(name, free=False)
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(PLATE_RADIUS, PLATE_HEIGHT / 2, 0), pos=(0, 0, PLATE_HEIGHT / 2),
                rgba=(0.95, 0.95, 0.93, 1.0), friction=(0.6, 5e-3, 5e-4))
  for x, y in DIRT:
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(DIRT_RADIUS, 0.0005, 0), pos=(x, y, PLATE_HEIGHT + 0.0005),
                  rgba=(0.45, 0.3, 0.15, 1.0), contype=0, conaffinity=0, mass=0.0)
  return spec
