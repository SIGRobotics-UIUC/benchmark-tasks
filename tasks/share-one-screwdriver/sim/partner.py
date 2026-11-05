"""The second arm's script, and the script the demonstration runs: an arm that waits its turn for the one
screwdriver, drives its own screw half a turn and stands the driver back.  It reads the scene's true state.

    arm = Driver(arm, ik, rng, dt)
    row = arm.act(env, e, now)       # every step: 6 joint targets (rad) and the gripper, 0 closed to 1 open

1. Wait a random while, then go for the driver only if it stands in its holder and the other hand is not at it;
   otherwise look again after a random while.  If, on the way, the other hand comes for it too, back off and wait.
2. Take it by the handle with the fingers straight down, lift it out and carry it, without turning the wrist, to
   2 cm over its own screw.
3. Turn the wrist so that the cross point lines up with the slot, as the two lie now; if the twist to come would take
   joint 6 past its range, a quarter turn back first.  Lower the point into the slot and twist a quarter turn
   clockwise; lift.  Again, until the screw is half a turn in.
4. Carry the driver back, fingers straight down over the holder, lower it in and let go."""

from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply

from sim import benchmark as b
from sim import props
from sim.env_cfg import READY_JOINTS
from sim.ik import ArmIk
from sim.robot import FINGER_MAX, GRASP_SITE
from sim.scene import LEFT_ARM, world_to_base
from typing import Callable

from sim.scene import BASE_POSE

START = (0.0, 4.0)
READY_Q = np.array(list(READY_JOINTS.values()))


def turn(axis, degrees: float) -> np.ndarray:
  q, m = np.zeros(4), np.zeros(9)
  import mujoco
  mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float), np.radians(degrees))
  mujoco.mju_quat2Mat(m, q)
  return m.reshape(3, 3)


def plumb(jaw: tuple[float, float, float]) -> np.ndarray:
  """Fingers straight down, the jaw (closing direction) along the horizontal ``jaw``."""
  y = np.asarray(jaw, dtype=float)
  z = np.array([0.0, 0.0, -1.0])
  return np.column_stack([np.cross(y, z), y, z])


def pos(env: ManagerBasedRlEnv, name: str, e: int) -> np.ndarray:
  return (env.scene[name].data.root_link_pos_w[e] - env.scene.env_origins[e]).cpu().numpy()


def site(env: ManagerBasedRlEnv, entity: str, name: str, e: int) -> np.ndarray:
  thing = env.scene[entity]
  return (thing.data.site_pos_w[e, thing.site_names.index(name)] - env.scene.env_origins[e]).cpu().numpy()


class Mover:
  """A queue of (joint targets, gripper) rows, refilled stage by stage."""

  def __init__(self, arm: str, ik: ArmIk, rng: np.random.Generator, dt: float) -> None:
    self.arm, self.ik, self.rng, self.dt = arm, ik, rng, dt
    self.side = 1.0 if arm == LEFT_ARM else -1.0
    self.q = READY_Q.copy()
    self.current = READY_Q.copy()
    self.grip = 1.0
    self.queue: list[tuple[np.ndarray, float]] = []
    self.rot = plumb((0.0, 1.0, 0.0))
    self.stage = "wait"
    self.start = rng.uniform(*START)
    self.log: list[str] = []
    self.failed = None

  def solve(self, point, through_table: bool = False) -> np.ndarray:
    q = self.ik.solve(world_to_base(self.arm, tuple(float(v) for v in point)), rot=self.rot, seed=self.q, starts=60,
                      iters=200, through_table=through_table)
    if q is None:
      raise RuntimeError(f"no IK at {np.round(point, 3).tolist()}")
    return q

  def glide(self, q: np.ndarray, seconds: float, grip: float) -> None:
    n = max(1, int(seconds / self.dt))
    for i in range(1, n + 1):
      s = i / n
      self.queue.append((self.q + (q - self.q) * (3 * s ** 2 - 2 * s ** 3), grip))
    self.q = q

  def line(self, a, b_, seconds: float, grip: float, through_table: bool = False) -> None:
    n = max(1, int(seconds / self.dt))
    for i in range(1, n + 1):
      self.q = self.solve(np.asarray(a) + (np.asarray(b_) - np.asarray(a)) * i / n, through_table)
      self.queue.append((self.q, grip))

  def hold(self, seconds: float, grip: float) -> None:
    self.queue += [(self.q, grip)] * max(1, int(seconds / self.dt))

  def plan(self, env: ManagerBasedRlEnv, e: int, now: float) -> None:
    raise NotImplementedError

  def act(self, env: ManagerBasedRlEnv, e: int, now: float) -> np.ndarray:
    if not self.queue and self.stage not in ("done", "failed"):
      try:
        self.plan(env, e, now)
      except RuntimeError as err:
        self.log.append(f"t={now:.1f} {self.stage}: {err}")
        self.failed = str(err)
        self.queue.clear()
        self.q = self.current.copy()
        self.hold(0.5, 1.0)
        self.glide(READY_Q, 1.2, 1.0)
        self.stage = "failed"
    if self.queue:
      self.current, self.grip = self.queue.pop(0)
    return np.r_[self.current, self.grip]


FIRST = (0.0, 3.0)
"""Seconds before it starts, drawn per episode."""
RETRY = (0.3, 1.5)
"""Seconds before it looks again, when the shared area was busy or it backed off."""
NEAR = 0.10
"""It leaves alone an object the other gripper is this close to."""
CARRY = 0.12
"""Height above the table it carries things at."""
LEAN = 10.0
"""Degrees the fingers lean away from the arm's base: held plumb, joint 4 is on its limit across most of the table."""
OPEN = 0.6
"""Gripper opening while reaching for a part: wide enough for a 3 cm part, narrow enough to miss its neighbours."""
SETTLE = 0.5
"""Seconds it waits at the bottom, fingers open, before closing."""


class Picker(Mover):
  def __init__(self, arm: str, ik, rng: np.random.Generator, dt: float,
               wanted: Callable[[ManagerBasedRlEnv, int], list[str]],
               drop: Callable[[ManagerBasedRlEnv, int, str], np.ndarray],
               busy: Callable[[np.ndarray], bool],
               grasp_height: Callable[[str], float] = lambda name: 0.0,
               square: Callable[[str], bool] = lambda name: True,
               drop_busy: Callable[[np.ndarray, np.ndarray], bool] = lambda other_site, spot: False,
               carry: float = CARRY, correct: bool = False, place_settle: float = 0.0,
               plumb_drop: bool = False, through: Callable[[np.ndarray], np.ndarray] | None = None,
               via_ready: bool = False, keep_jaw: bool = False, approach: float = 0.10) -> None:
    super().__init__(arm, ik, rng, dt)
    self.wanted, self.drop, self.busy, self.grasp_height = wanted, drop, busy, grasp_height
    self.target_square, self.drop_busy, self.carry = square, drop_busy, carry
    # ``correct``: stop over the drop, see where the part sits in the hand and lower it so the part, not the hand, lands
    # on the spot.  ``place_settle``: seconds to wait at the bottom, still holding, before letting go.
    self.correct, self.place_settle = correct, place_settle
    # ``plumb_drop``: let go with the fingers straight down wherever that is within reach, at any height: a leaning
    # hand, opening, pushes the part sideways.
    self.plumb_drop = plumb_drop
    # ``through(spot)``: a point on this side of an opening the part must pass through to reach ``spot``.  The part is
    # carried to that point, then straight through the opening to the drop, and the hand comes back the same way.
    self.through = through
    # ``via_ready``: after a drop, go to the next part by way of ready, high and clear, not straight across in joint
    # space, which can sweep the hand low through what was just put down.
    self.via_ready = via_ready
    # ``keep_jaw``: carry the part without turning the wrist, but for a half turn, so that it is put down lying the way
    # it lay when picked.
    self.keep_jaw = keep_jaw
    self.approach = approach
    """How far above the part the hand starts its straight descent, and lifts it to."""
    self.release_rows = 0
    self.other = "right_arm" if arm == "left_arm" else "left_arm"
    self.start = rng.uniform(*FIRST)
    self.target = None
    self.placed: list[str] = []
    self.backoffs = 0

  def _jaw(self, env: ManagerBasedRlEnv, e: int, name: str) -> tuple[float, float, float]:
    """Across the object's shorter side, turned to suit this arm."""
    q = env.scene[name].data.root_link_quat_w[e:e + 1]
    x = quat_apply(q, q.new_tensor([[1.0, 0.0, 0.0]]))[0].cpu().numpy()
    y = quat_apply(q, q.new_tensor([[0.0, 1.0, 0.0]]))[0].cpu().numpy()
    # Along the object's x: across a bar whose length is its y; any way does for a cube or an upright cylinder.
    jaw = x[:2] / max(np.linalg.norm(x[:2]), 1e-6)
    if abs(y[2]) > 0.7:  # Its long side stands up: any horizontal jaw.
      jaw = np.array([1.0, 0.0])
    if jaw[0] * self.side < 0:
      jaw = -jaw
    return (float(jaw[0]), float(jaw[1]), 0.0)

  def _long_axis(self, env: ManagerBasedRlEnv, e: int, name: str) -> np.ndarray:
    q = env.scene[name].data.root_link_quat_w[e:e + 1]
    return quat_apply(q, q.new_tensor([[0.0, 1.0, 0.0]]))[0].cpu().numpy()

  def lean_rot(self, jaw, point, lean: float) -> np.ndarray:
    """Fingers down, the jaw along ``jaw``, leaning ``lean`` degrees about the jaw, the fingertips away from the base."""
    base = np.array(BASE_POSE[self.arm][0][:2])
    radial = np.asarray(point[:2]) - base
    best = None
    for sign in (1.0, -1.0):
      rot = turn(jaw, sign * lean) @ plumb(jaw)
      score = float(rot[:2, 2] @ radial)
      if best is None or score > best[0]:
        best = (score, rot)
    return best[1]

  def holding(self, env: ManagerBasedRlEnv, e: int) -> bool:
    if self.target is None:
      return False
    robot = env.scene[self.arm]
    fingers = float(robot.data.joint_pos[e, robot.joint_names.index("joint7")])
    return np.linalg.norm(pos(env, self.target, e) - site(env, self.arm, GRASP_SITE, e)) < 0.04 and fingers < 0.8 * FINGER_MAX

  def act(self, env: ManagerBasedRlEnv, e: int, now: float) -> np.ndarray:
    if self.stage == "approach" and self.queue and self.busy(site(env, self.other, GRASP_SITE, e)):
      # The other gripper came into the shared area on our way in: back off from where we are, try again later.
      self.queue.clear()
      self.q = self.current.copy()
      self.glide(READY_Q, 0.8, 1.0)
      self.stage, self.backoffs = "backoff", self.backoffs + 1
    elif ((self.stage == "over" or self.stage == "placed" and len(self.queue) > self.release_rows) and self.queue
          and self.drop_busy(site(env, self.other, GRASP_SITE, e), self.spot)):
      # The other gripper came to the drop on our way there: take the part back to where it was lifted, try again.
      self.queue.clear()
      self.q = self.current.copy()
      self.rot = self.lift_rot
      self.glide(self.lift_q, 1.0, 0.0)
      self.stage, self.backoffs = "lifted", self.backoffs + 1
    return super().act(env, e, now)

  def _plan_through(self, spot: np.ndarray, choices: list[np.ndarray]) -> None:
    """Carry the part to the near side of the opening, through it to the drop, let go, and come back out."""
    before, low = self.through(spot), spot + [0, 0, 0.02]
    ways = []
    for rot in choices:
      self.rot = rot
      try:
        q = self.solve(before)
      except RuntimeError:
        continue
      ways.append((float(np.abs(q - self.q).max()), rot, q))
    start_q, start_len = self.q.copy(), len(self.queue)
    for _, rot, q in sorted(ways, key=lambda w: w[0]):
      self.rot = rot
      try:
        self.glide(q, 2.0, 0.0)
        self.line(before, low, 1.0, 0.0)
        if self.place_settle:
          self.hold(self.place_settle, 0.0)
        mark = len(self.queue)
        self.hold(0.4, 1.0)
        self.line(low, before, 0.8, 1.0)
        self.line(before, before + [0, 0, 0.08], 0.5, 1.0)  # Up, empty-handed, clear of what is on the table.
      except RuntimeError:
        del self.queue[start_len:]
        self.q = start_q.copy()
        continue
      self.release_rows = len(self.queue) - mark
      self.stage = "placed"
      return
    raise RuntimeError(f"no IK through to {np.round(spot, 3).tolist()}")

  def plan(self, env: ManagerBasedRlEnv, e: int, now: float) -> None:
    if self.stage in ("wait", "backoff"):
      if self.stage == "backoff":
        self.start, self.stage = now + self.rng.uniform(*RETRY), "wait"
        return
      if now < self.start:
        return
      wanted = self.wanted(env, e)
      if not wanted:
        self.glide(READY_Q, 1.2, 1.0)
        self.stage = "finishing"
        return
      other = site(env, self.other, GRASP_SITE, e)
      free = [n for n in wanted if np.linalg.norm(pos(env, n, e)[:2] - other[:2]) > NEAR]
      if self.busy(other) or not free:
        self.start = now + self.rng.uniform(*RETRY)
        return
      # Nearest to this arm's side first.
      self.target = max(free, key=lambda n: self.side * pos(env, n, e)[1])
      self.grasp = pos(env, self.target, e) + [0.0, 0.0, self.grasp_height(self.target)]
      jaw = self._jaw(env, e, self.target)
      # Leaning about the jaw keeps the pads upright and square to the part; a turn of the jaw a quarter turn suits a
      # cube or an upright cylinder as well, and may be within reach where the first is not.
      flip = (-jaw[0], -jaw[1], 0.0)  # The same grip with the wrist turned half round.
      options = [(jaw, LEAN), (jaw, 2 * LEAN), (flip, LEAN), (flip, 2 * LEAN), (jaw, 3 * LEAN), (flip, 3 * LEAN)]
      if self.plumb_drop:  # Picked plumb, it is let go plumb without the wrist turning in between: a turn tilts it.
        quarter = (-jaw[1], jaw[0], 0.0)
        options = [(jaw, 0.0), (flip, 0.0), (quarter, 0.0), (tuple(-v for v in quarter), 0.0)] + options
      if abs(self._long_axis(env, e, self.target)[2]) > 0.7 or self.target_square(self.target):
        quarter = np.array([-jaw[1], jaw[0]])
        if quarter[0] * self.side < 0 or (quarter[0] == 0 and quarter[1] < 0):
          quarter = -quarter
        options += [((float(quarter[0]), float(quarter[1]), 0.0), LEAN), ((float(quarter[0]), float(quarter[1]), 0.0), 2 * LEAN)]
      for jaw, lean in options:
        self.rot = self.lean_rot(jaw, self.grasp, lean)
        try:
          q = self.solve(self.grasp + [0, 0, self.approach])
          self.solve(self.grasp)
        except RuntimeError:
          continue
        self.jaw, self.lean = jaw, lean
        if self.via_ready and self.placed:
          self.glide(READY_Q, 1.0, OPEN)
        self.glide(q, 1.2, OPEN)
        self.stage = "approach"
        return
      self.log.append(f"t={now:.1f}: cannot reach {self.target}")
      self.start, self.target = now + self.rng.uniform(*RETRY), None
    elif self.stage == "approach":
      self.line(self.grasp + [0, 0, self.approach], self.grasp, 0.8, OPEN)
      self.hold(SETTLE, OPEN)  # The arm lags its targets: let it get all the way down before closing.
      self.hold(0.6, 0.0)
      self.line(self.grasp, self.grasp + [0, 0, self.approach], 0.6, 0.0)
      self.stage = "lifted"
    elif self.stage == "lifted":
      if not self.holding(env, e):
        self.log.append(f"t={now:.1f}: missed {self.target} at {np.round(pos(env, self.target, e), 3).tolist()}")
        self.glide(READY_Q, 1.0, 1.0)
        self.stage, self.target = "backoff", None
        return
      spot = self.spot = self.drop(env, e, self.target)
      if self.drop_busy(site(env, self.other, GRASP_SITE, e), spot):
        self.hold(self.rng.uniform(*RETRY), 0.0)  # Wait, holding the part where it is, and look again.
        return
      self.lift_q, self.lift_rot = self.q.copy(), self.rot
      here = site(env, self.arm, GRASP_SITE, e)
      # Carry it the way it was picked if that reaches the drop, else whichever lean does with the least turn of the
      # joints: a big swing of the wrist on the way flings the part out of the fingers.  As high as it reaches.
      flip = (-self.jaw[0], -self.jaw[1], 0.0)
      # Held, the part turns with the hand: past the grip it was picked with, any turn of the wrist will do, and over a
      # far drop only a quarter of the turns are within reach.
      turns = [] if self.keep_jaw else [np.radians(a) for a in (90, -90, 45, -45, 135, -135)]
      jaws = [self.jaw, flip] + [(self.jaw[0] * np.cos(a) - self.jaw[1] * np.sin(a),
                                  self.jaw[0] * np.sin(a) + self.jaw[1] * np.cos(a), 0.0) for a in turns]
      choices = [self.rot] + [self.lean_rot(jaw, spot, lean) for jaw in jaws for lean in (LEAN, 2 * LEAN, 0.0, 3 * LEAN)]
      if self.through is not None:
        self._plan_through(spot, choices)
        return
      top = max(here[2], 0.75 + self.carry)
      ways = []  # (swing, height, rotation, joints at the top): from the highest height any of them reaches.
      heights = np.arange(top, spot[2] + 0.019, -0.02)
      for z in heights:
        for rot in choices:
          self.rot = rot
          try:
            q = self.solve(np.r_[spot[:2], z])
          except RuntimeError:
            continue
          ways.append((float(np.abs(q - self.q).max()), z, rot, q))
        if ways:
          break
      # ``plumb_drop``: having carried it high, turn the fingers straight down just over the drop, then lower.  Only the
      # few plumb turns of the wrist are tried there: a failed solve is slow.
      plumbs = [self.lean_rot(jaw, spot, 0.0) for jaw in jaws] if self.plumb_drop else []
      hover = spot + [0, 0, 0.05]
      # The solver starts from random guesses, and near the edge of reach a pose found once may not be found again on
      # the way down: build the whole move for each way, least swing first, and take the first that holds together.
      start_q, start_len = self.q.copy(), len(self.queue)
      for _, z, rot, q in sorted(ways, key=lambda w: w[0]):
        high = np.r_[spot[:2], z]
        self.rot = rot
        try:
          self.glide(q, 2.0, 0.0)
          if plumbs:
            for plumb_rot in plumbs:
              self.rot = plumb_rot
              try:
                q_hover = self.solve(hover)
                break
              except RuntimeError:
                continue
            else:
              raise RuntimeError("no plumb pose over the drop")
            self.glide(q_hover, 1.0, 0.0)
            high = hover
          after_glide, q_high = len(self.queue), self.q.copy()
          self.line(high, spot + [0, 0, 0.02], 0.6, 0.0)
          if self.place_settle:
            self.hold(self.place_settle, 0.0)
          before = len(self.queue)
          self.hold(0.4, 1.0)
          self.line(spot + [0, 0, 0.02], high, 0.5, 1.0)
        except RuntimeError:
          del self.queue[start_len:]
          self.q = start_q.copy()
          continue
        self.release_rows = len(self.queue) - before  # Once it starts letting go, it no longer turns back.
        self.stage = "placed"
        if self.correct:  # Only as far as over the drop; the rest is planned there, from where the part sits.
          del self.queue[after_glide:]
          self.q, self.high, self.stage = q_high, high, "over"
        return
      raise RuntimeError(f"no IK at {np.round(np.r_[spot[:2], top], 3).tolist()}")
    elif self.stage == "over":
      if not self.holding(env, e):
        self.log.append(f"t={now:.1f}: dropped {self.target} on the way")
        self.glide(READY_Q, 1.0, 1.0)
        self.stage, self.target = "backoff", None
        return
      if self.drop_busy(site(env, self.other, GRASP_SITE, e), self.spot):  # Theirs now: back to where it lifted.
        self.rot = self.lift_rot
        self.glide(self.lift_q, 1.0, 0.0)
        self.stage, self.backoffs = "lifted", self.backoffs + 1
        return
      # Where the part sits in the hand now, height too: it may have slipped since it was picked up.  The drop spot
      # was planned for the part held at its grasp height.
      offset = pos(env, self.target, e) - site(env, self.arm, GRASP_SITE, e)
      planned = np.array([0.0, 0.0, -self.grasp_height(self.target)])
      start_q, start_len = self.q.copy(), len(self.queue)
      for shift in (offset - planned, np.zeros(3)):
        low = self.spot + [0, 0, 0.02] - shift
        try:
          self.line(self.high, low, 0.8, 0.0)
          if self.place_settle:
            self.hold(self.place_settle, 0.0)
          before = len(self.queue)
          self.hold(0.4, 1.0)
          self.line(low, self.high, 0.5, 1.0)
        except RuntimeError:
          del self.queue[start_len:]
          self.q = start_q.copy()
          continue
        self.release_rows = len(self.queue) - before
        self.stage = "placed"
        return
      raise RuntimeError(f"no IK down to {np.round(self.spot, 3).tolist()}")
    elif self.stage == "placed":
      off = np.linalg.norm(pos(env, self.target, e)[:2] - self.spot[:2])
      self.placed.append(self.target)
      self.log.append(f"t={now:.1f}: put {self.target} down" + ("" if off < 0.05 else f", {off:.2f} m from where meant"))
      self.target = None
      self.stage = "wait"
      self.start = now
    elif self.stage == "finishing":
      self.stage = "done"



def yaw(env, name: str, e: int) -> float:
  """The heading of ``name``'s x axis, about the vertical."""
  q = env.scene[name].data.root_link_quat_w[e:e + 1]
  x = quat_apply(q, q.new_tensor([[1.0, 0.0, 0.0]]))[0].cpu().numpy()
  return float(np.arctan2(x[1], x[0]))

def screw_angle(env, arm: str, e: int) -> float:
  return float(env.scene[b.SCREWS[arm]].data.joint_pos[e, 0])

class Driver(Picker):
  """A Picker for the one screwdriver: takes it from its holder when it stands there and the other hand is not at it,
  drives its own screw half a turn, and stands the driver back in its holder."""

  TWIST = np.pi / 2
  """Each twist of the wrist: two make the half turn.  Between them it lifts, turns back and seats the point again: the
  cross slot is the same a quarter turn round."""
  TWIST_S = 1.5
  SEAT = 0.001
  SPIN_SIGN = -1.0
  MAX_TWISTS = 5
  """Which way a turn of joint 6 turns the tool about the vertical: +1 if positive is anticlockwise from above."""
  """How far into the slot's floor the tip is aimed, to keep it seated."""

  def __init__(self, arm: str, ik, rng, dt) -> None:
    holder = np.array(b.DRIVER_XY)
    self.driven = False
    self.home_z = b.Z + b.driver_rest_z()

    def wanted(env, e):
      p = pos(env, "driver", e)
      standing = np.linalg.norm(p[:2] - holder) < 0.015 and p[2] > self.home_z - 0.01
      return [] if self.driven or not standing else ["driver"]

    def drop(env, e, name):
      # Its handle 1 cm over the holder's top: let go and it drops onto it.
      return np.r_[holder, self.home_z + 0.01 - 0.02]

    def busy(other_site):
      return np.linalg.norm(other_site[:2] - holder) < 0.14 and other_site[2] < b.Z + 0.30

    super().__init__(arm, ik, rng, dt, wanted, drop, busy, square=lambda name: True, plumb_drop=True, keep_jaw=True, grasp_height=lambda name: -0.012,
                     carry=0.13, approach=0.05)

  def plan(self, env, e: int, now: float) -> None:
    if self.stage == "wait" and not self.driven and now >= self.start and not self.wanted(env, e):
      self.start = now + self.rng.uniform(*RETRY)  # The other arm has the driver: look again later.
      return
    if self.stage == "lifted" and not self.driven and self.holding(env, e):
      self._plan_drive(env, e, now)
      return
    if self.stage == "seat":
      self._plan_seat(env, e, now)
      return
    super().plan(env, e, now)

  def _tip_over_screw(self, env, e: int, height: float) -> np.ndarray:
    """Where the grasp site goes for the driver's tip to be ``height`` over the screw's slot floor, from where the
    driver sits in the hand now."""
    offset = pos(env, "driver", e) - site(env, self.arm, GRASP_SITE, e)
    floor = np.r_[b.SCREW_XY[self.arm], b.Z + props.SCREW_BLOCK[2]]
    return floor + [0, 0, props.DRIVER_TIP + height] - offset

  def _plan_drive(self, env, e: int, now: float) -> None:
    """Carry the driver plumb, without turning the wrist, to 4 cm over the screw."""
    self.rot = self.lean_rot(self.jaw, np.r_[b.SCREW_XY[self.arm], 0.0], 0.0)
    hover = self._tip_over_screw(env, e, 0.02)
    self.lift_q, self.lift_rot = self.q.copy(), self.rot
    self.glide(self.solve(hover + [0, 0, 0.02]), 2.0, 0.0)
    self.line(hover + [0, 0, 0.02], hover, 0.4, 0.0)  # Plumb, the YAM reaches only some 13 cm up here.
    self.hover = hover
    self.stage = "seat"
    self.twists = 0

  def _plan_seat(self, env, e: int, now: float) -> None:
    """Over the screw: turn the wrist so the cross point lines up with the slot, lower it in, and twist."""
    if not self.holding(env, e):
      self.log.append(f"t={now:.1f}: dropped the driver")
      self.glide(READY_Q, 1.0, 1.0)
      self.stage, self.target, self.driven = "backoff", None, True
      return
    if screw_angle(env, self.arm, e) <= -props.SCREW_TURN + 0.05 or self.twists >= self.MAX_TWISTS:
      self.driven = True
      self.stage = "lifted"  # Home, or as far as it goes: now the Picker takes the driver back to its holder.
      return
    # The slot and the point are each alike a quarter turn round: the smallest turn that lines them up.
    off = (yaw(env, "driver", e) - screw_angle(env, self.arm, e) + np.pi / 4) % (np.pi / 2) - np.pi / 4
    q = self.q.copy()
    q[5] -= self.SPIN_SIGN * off  # Turns the driver by -off about the vertical.
    # Room for the twist within joint 6's range: if not, a quarter turn back first, which the slot does not mind.
    twist = self.SPIN_SIGN * -self.TWIST
    lo, hi = self.ik.lower[5] + 0.05, self.ik.upper[5] - 0.05
    while not lo <= q[5] + twist <= hi:
      q[5] -= np.sign(twist) * np.pi / 2
    self.queue += [(q, 0.0)] * max(1, int(0.8 / self.dt))
    self.q = q
    seat = self._tip_over_screw(env, e, -self.SEAT)
    up = self._tip_over_screw(env, e, 0.02)
    self.rot = self.fk_rot(q)  # Plumb, turned as the wrist now is: the bases are not turned, so base frame is world.
    self.line(up, seat, 0.6, 0.0)
    self.hold(0.3, 0.0)
    for k in range(int(self.TWIST_S / self.dt)):
      q = self.q.copy()
      q[5] += self.SPIN_SIGN * -self.TWIST / int(self.TWIST_S / self.dt)  # Clockwise, from above: in.
      self.queue.append((q, 0.0))
      self.q = q
    self.rot = self.fk_rot(self.q)
    self.line(seat, up, 0.5, 0.0)
    self.twists += 1
    self.log.append(f"t={now:.1f}: twist {self.twists}, screw at {np.degrees(screw_angle(env, self.arm, e)):.0f} deg")

  def fk_rot(self, q: np.ndarray) -> np.ndarray:
    return self.ik.rotation(q)
