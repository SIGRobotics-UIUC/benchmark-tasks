"""The objects: a drying rack of one-cup bays, two mats and the cups.

Each function returns an ``MjSpec`` with one root body.  Props that stay put have no free joint; the cups have one."""

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
RACK_BAYS = 4
RACK_BAY = 0.05
"""Inside width of a bay, both ways: a 3.4 cm cup stands in it, two do not fit."""
RACK_WALL = 0.006
RACK_HEIGHT = 0.05
RACK_FLOOR = 0.006


def rack_bay_x(i: int) -> float:
  """The middle of bay ``i`` along the rack (its x), from the rack's middle; bay 0 is nearest the arms."""
  return (i - (RACK_BAYS - 1) / 2) * (RACK_BAY + RACK_WALL)


def drying_rack_spec() -> mujoco.MjSpec:
  """A row of square bays along x, open at the top: walls, a floor and a divider between bays."""
  spec, body = _root("drying_rack", free=False)
  rgba = (0.85, 0.85, 0.88, 1.0)
  length = RACK_BAYS * RACK_BAY + (RACK_BAYS - 1) * RACK_WALL
  _open_box(body, (length, RACK_BAY), RACK_HEIGHT, wall=RACK_WALL, floor=RACK_FLOOR, rgba=rgba)
  for i in range(RACK_BAYS - 1):
    x = (rack_bay_x(i) + rack_bay_x(i + 1)) / 2
    body.add_geom(type=BOX, size=(RACK_WALL / 2, RACK_BAY / 2, RACK_HEIGHT / 2), pos=(x, 0, RACK_FLOOR + RACK_HEIGHT / 2),
                  rgba=rgba, friction=GRIP_FRICTION)
  return spec
