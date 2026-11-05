"""The second arm's script, and the script the demonstration runs: an arm that fills its own kit from the shared
parts tray, keeping out of the other arm's way.  It reads the scene's true state.

    arm = Kitter(arm, ik, rng, dt)
    row = arm.act(env, e, now)       # every step: 6 joint targets (rad) and the gripper, 0 closed to 1 open

1. Wait a random while.  Then, while its kit lacks something: if the other gripper is down over the parts tray, look
   again after a random while; else go for the part it needs nearest its own side that the other gripper is not
   next to.  If the other gripper comes down over the tray while it is on its way in, back off and wait a while.
2. Come down over the part, fingers straight down but for a slight lean, across its narrower side; close and lift.
3. Carry it high to the next free corner of its kit tray, lower it and let go.
4. When the kit is complete, go back to ready."""

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
               carry: float = CARRY) -> None:
    super().__init__(arm, ik, rng, dt)
    self.wanted, self.drop, self.busy, self.grasp_height = wanted, drop, busy, grasp_height
    self.target_square, self.drop_busy, self.carry = square, drop_busy, carry
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
    elif (self.stage == "placed" and len(self.queue) > self.release_rows
          and self.drop_busy(site(env, self.other, GRASP_SITE, e), self.spot)):
      # The other gripper came to the drop on our way there: take the part back to where it was lifted, try again.
      self.queue.clear()
      self.q = self.current.copy()
      self.rot = self.lift_rot
      self.glide(self.lift_q, 1.0, 0.0)
      self.stage, self.backoffs = "lifted", self.backoffs + 1
    return super().act(env, e, now)

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
      if abs(self._long_axis(env, e, self.target)[2]) > 0.7 or self.target_square(self.target):
        quarter = np.array([-jaw[1], jaw[0]])
        if quarter[0] * self.side < 0 or (quarter[0] == 0 and quarter[1] < 0):
          quarter = -quarter
        options += [((float(quarter[0]), float(quarter[1]), 0.0), LEAN), ((float(quarter[0]), float(quarter[1]), 0.0), 2 * LEAN)]
      for jaw, lean in options:
        self.rot = self.lean_rot(jaw, self.grasp, lean)
        try:
          q = self.solve(self.grasp + [0, 0, 0.10])
          self.solve(self.grasp)
        except RuntimeError:
          continue
        self.jaw, self.lean = jaw, lean
        self.glide(q, 1.2, OPEN)
        self.stage = "approach"
        return
      self.log.append(f"t={now:.1f}: cannot reach {self.target}")
      self.start, self.target = now + self.rng.uniform(*RETRY), None
    elif self.stage == "approach":
      self.line(self.grasp + [0, 0, 0.10], self.grasp, 0.8, OPEN)
      self.hold(SETTLE, OPEN)  # The arm lags its targets: let it get all the way down before closing.
      self.hold(0.6, 0.0)
      self.line(self.grasp, self.grasp + [0, 0, 0.10], 0.6, 0.0)
      self.stage = "lifted"
    elif self.stage == "lifted":
      if not self.holding(env, e):
        self.log.append(f"t={now:.1f}: missed {self.target}")
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
      turns = [np.radians(a) for a in (90, -90, 45, -45, 135, -135)]
      jaws = [self.jaw, flip] + [(self.jaw[0] * np.cos(a) - self.jaw[1] * np.sin(a),
                                  self.jaw[0] * np.sin(a) + self.jaw[1] * np.cos(a), 0.0) for a in turns]
      choices = [self.rot] + [self.lean_rot(jaw, spot, lean) for jaw in jaws for lean in (LEAN, 2 * LEAN, 0.0, 3 * LEAN)]
      top = max(here[2], 0.75 + self.carry)
      ways = []  # (swing, height, rotation, joints at the top): from the highest height any of them reaches.
      for z in np.arange(top, spot[2] + 0.019, -0.02):
        for rot in choices:
          self.rot = rot
          try:
            q = self.solve(np.r_[spot[:2], z])
          except RuntimeError:
            continue
          ways.append((float(np.abs(q - self.q).max()), z, rot, q))
        if ways:
          break
      # The solver starts from random guesses, and near the edge of reach a pose found once may not be found again on
      # the way down: build the whole move for each way, least swing first, and take the first that holds together.
      start_q, start_len = self.q.copy(), len(self.queue)
      for _, z, rot, q in sorted(ways, key=lambda w: w[0]):
        high = np.r_[spot[:2], z]
        self.rot = rot
        try:
          self.glide(q, 2.0, 0.0)
          self.line(high, spot + [0, 0, 0.02], 0.6, 0.0)
          before = len(self.queue)
          self.hold(0.4, 1.0)
          self.line(spot + [0, 0, 0.02], high, 0.5, 1.0)
        except RuntimeError:
          del self.queue[start_len:]
          self.q = start_q.copy()
          continue
        self.release_rows = len(self.queue) - before  # Once it starts letting go, it no longer turns back.
        self.stage = "placed"
        return
      raise RuntimeError(f"no IK at {np.round(np.r_[spot[:2], top], 3).tolist()}")
    elif self.stage == "placed":
      off = np.linalg.norm(pos(env, self.target, e)[:2] - self.spot[:2])
      self.placed.append(self.target)
      self.log.append(f"t={now:.1f}: put {self.target} down" + ("" if off < 0.05 else f", {off:.2f} m from where meant"))
      self.target = None
      self.stage = "wait"
      self.start = now
    elif self.stage == "finishing":
      self.stage = "done"


def in_kit(env: ManagerBasedRlEnv, arm: str, e: int) -> dict[str, bool]:
  """Per part, whether it rests inside ``arm``'s kit tray."""
  centre, half = np.array(b.KIT_TRAY_XY[arm]), np.array(b.KIT_TRAY_INNER) / 2
  return {n: bool((np.abs(pos(env, n, e)[:2] - centre) < half).all() and pos(env, n, e)[2] < b.Z + 0.06) for n in b.PARTS}


def Kitter(arm: str, ik, rng: np.random.Generator, dt: float) -> Picker:
  """A Picker that fills ``arm``'s kit with its recipe from the shared tray."""
  recipe = b.RECIPES[arm]
  other = "right_arm" if arm == "left_arm" else "left_arm"
  centre = np.array(b.PARTS_TRAY_XY)

  def wanted(env, e):
    inside, theirs = in_kit(env, arm, e), in_kit(env, other, e)
    have = {k: sum(inside[n] for n, (kind, _) in b.PARTS.items() if kind == k) for k in recipe}
    # Anywhere on the table but in a kit: one knocked out of the parts tray is still fetched.
    return [n for n, (kind, _) in b.PARTS.items()
            if have.get(kind, 0) < recipe.get(kind, 0) and not inside[n] and not theirs[n]
            and abs(pos(env, n, e)[1]) < 0.42 and b.Z < pos(env, n, e)[2] < b.Z + 0.08]

  def drop(env, e, name):
    used = sum(in_kit(env, arm, e).values())
    dx, dy = [(-0.05, -0.04), (0.05, -0.04), (-0.05, 0.04), (0.05, 0.04)][min(used, 3)]
    kx, ky = b.KIT_TRAY_XY[arm]
    return np.array([kx + dx, ky + dy, b.Z + 0.004 + props.part_half_height(b.PARTS[name][0]) + 0.006])

  def busy(other_site):
    """The other gripper is over the parts tray, low enough to be working in it."""
    return abs(other_site[0] - centre[0]) < 0.20 and abs(other_site[1]) < 0.17 and other_site[2] < b.Z + 0.16

  return Picker(arm, ik, rng, dt, wanted, drop, busy, grasp_height=lambda name: 0.003,
                square=lambda name: b.PARTS[name][0] != "yellow")
