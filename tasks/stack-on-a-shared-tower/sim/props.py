"""The objects: a tower base, two mats and the blocks.

Each function returns an ``MjSpec`` with one root body.  Props that stay put have no free joint; the blocks have one."""

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
PART_KINDS = {
  "red": {"shape": "cube", "size": 0.015, "mass": 0.08, "rgba": (0.8, 0.15, 0.12, 1.0)},
  "blue": {"shape": "cylinder", "size": (0.014, 0.0175), "mass": 0.07, "rgba": (0.15, 0.3, 0.8, 1.0)},
  "yellow": {"shape": "bar", "size": (0.015, 0.026, 0.012), "mass": 0.08, "rgba": (0.95, 0.8, 0.15, 1.0)},
  # Rubbish, for the trash can: shaped like the parts, so that one script picks up either.
  "paper": {"shape": "cube", "size": 0.016, "mass": 0.06, "rgba": (0.93, 0.93, 0.9, 1.0)},
  "tin": {"shape": "cylinder", "size": (0.015, 0.02), "mass": 0.06, "rgba": (0.72, 0.74, 0.78, 1.0)},
  "wrapper": {"shape": "bar", "size": (0.015, 0.026, 0.012), "mass": 0.07, "rgba": (0.95, 0.45, 0.1, 1.0)},
  # Cups, for the drying rack: one colour per arm.
  "green_cup": {"shape": "cylinder", "size": (0.017, 0.015), "mass": 0.07, "rgba": (0.3, 0.7, 0.35, 1.0)},
  "blue_cup": {"shape": "cylinder", "size": (0.017, 0.015), "mass": 0.07, "rgba": (0.25, 0.4, 0.85, 1.0)},
  # Blocks, for the shared tower: 4 x 4 x 2 cm slabs, one colour per arm.
  "green_block": {"shape": "bar", "size": (0.02, 0.02, 0.01), "mass": 0.1, "rgba": (0.3, 0.7, 0.35, 1.0)},
  "blue_block": {"shape": "bar", "size": (0.02, 0.02, 0.01), "mass": 0.1, "rgba": (0.25, 0.4, 0.85, 1.0)},
}
"""The three kinds of part: a 3 cm red cube (a bracket), a blue cylinder 2.8 cm across and 3.5 cm tall standing on end
(a bushing), and a yellow 3 x 5.2 x 2.4 cm block lying flat, its length along y (a spacer).  Half sizes, metres."""
PART_FRICTION = (1.0, 0.02, 5e-4)
"""Torsional friction (condim 4) so a part squeezed off-centre does not spin in the fingers."""


def part_spec(name: str, kind: str) -> mujoco.MjSpec:
  """One part, origin at its middle."""
  spec, body = _root(name, free=True)
  k = PART_KINDS[kind]
  if k["shape"] == "cube":
    body.add_geom(type=BOX, size=(k["size"],) * 3, mass=k["mass"], rgba=k["rgba"], friction=PART_FRICTION, condim=4)
  elif k["shape"] == "cylinder":
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(k["size"][0], k["size"][1], 0), mass=k["mass"],
                  rgba=k["rgba"], friction=PART_FRICTION, condim=4)
  else:
    body.add_geom(type=BOX, size=k["size"], mass=k["mass"], rgba=k["rgba"], friction=PART_FRICTION, condim=4)
  return _stiffen(spec)


def part_half_height(kind: str) -> float:
  k = PART_KINDS[kind]
  return k["size"] if k["shape"] == "cube" else k["size"][1] if k["shape"] == "cylinder" else k["size"][2]
DESK_MAT = (0.20, 0.20)


def desk_mat_spec(name: str, rgba) -> mujoco.MjSpec:
  """A flat mat marking one arm's desk: 3 mm thick, nothing to catch on."""
  spec, body = _root(name, free=False)
  body.add_geom(type=BOX, size=(DESK_MAT[0] / 2, DESK_MAT[1] / 2, 0.0015), pos=(0, 0, 0.0015), rgba=rgba, friction=GRIP_FRICTION)
  return spec
TOWER_BASE = (0.06, 0.06, 0.02)
"""The tower's base: a 6 x 6 cm block 2 cm tall, fixed to the table; the first block goes on it."""


def tower_base_spec() -> mujoco.MjSpec:
  spec, body = _root("tower_base", free=False)
  body.add_geom(type=BOX, size=(TOWER_BASE[0] / 2, TOWER_BASE[1] / 2, TOWER_BASE[2] / 2), pos=(0, 0, TOWER_BASE[2] / 2),
                rgba=(0.55, 0.4, 0.25, 1.0), friction=PART_FRICTION, condim=4)
  return _stiffen(spec)
