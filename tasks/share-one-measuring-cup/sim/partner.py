"""The second arm's script, and the script the demonstration runs: an arm that waits its turn for the one cup,
pours its share into its own bowl and stands the cup back.  It reads the scene's true state.

    arm = Pourer(arm, ik, rng, dt)
    row = arm.act(env, e, now)       # every step: 6 joint targets (rad) and the gripper, 0 closed to 1 open

1. Wait a random while, then go for the cup only if it stands where it started and the other hand is not at it;
   otherwise look again after a random while.
2. Take it by its two tabs with the fingers straight down and carry it upright to just off its own bowl.
3. Tilt it about the level line across its grip, 10 degrees at a time, then from 40 degrees 2.5 at a time.  Watch
   every step: once 6 beads have left the cup, tilt back upright at once.
4. Stand the cup back where it started and let go."""

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
               via_ready: bool = False, keep_jaw: bool = False, approach: float = 0.10, opening: float = OPEN) -> None:
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
    self.opening = opening
    """Gripper opening while reaching for a part."""
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
          self.glide(READY_Q, 1.0, self.opening)
        self.glide(q, 1.2, self.opening)
        self.stage = "approach"
        return
      self.log.append(f"t={now:.1f}: cannot reach {self.target}")
      self.start, self.target = now + self.rng.uniform(*RETRY), None
    elif self.stage == "approach":
      self.line(self.grasp + [0, 0, self.approach], self.grasp, 0.8, self.opening)
      self.hold(SETTLE, self.opening)  # The arm lags its targets: let it get all the way down before closing.
      self.hold(0.6, 0.0)
      self.line(self.grasp, self.grasp + [0, 0, self.approach], 0.6, 0.0)
      self.stage = "lifted"
    elif self.stage == "lifted":
      if not self.holding(env, e):
        self.log.append(f"t={now:.1f}: missed {self.target} at {np.round(pos(env, self.target, e), 3).tolist()}, hand at {np.round(site(env, self.arm, GRASP_SITE, e), 3).tolist()}")
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



def beads_in_bowl(env: ManagerBasedRlEnv, arm: str) -> np.ndarray:
  """Per env, how many beads lie in ``arm``'s bowl."""
  centre = np.array(b.MBOWL_XY[arm])
  out = np.zeros(env.num_envs)
  for e in range(env.num_envs):
    for n in b.BEADS:
      p = pos(env, n, e)
      out[e] += bool((np.abs(p[:2] - centre) < b.MBOWL_INNER / 2).all() and p[2] < b.Z + b.MBOWL_HEIGHT + 0.01)
  return out


class Pourer(Picker):
  """A Picker for the one cup of beads: takes it when it stands where it started and the other hand is not at it,
  tilts it over its own bowl a little at a time until its share is in, and stands the cup back."""

  TARGET = 6
  """Beads it pours: the low end of a share of 6 to 10; those already on their way when it stops make up the rest."""
  STEP = 10.0
  """Degrees of tilt at a time, waiting between for the beads to settle..."""
  SLOW_FROM = 40.0
  FINE = 2.5
  """...and past SLOW_FROM, where the beads begin to go, FINE degrees at a time, waiting longer."""
  MAX_TILT = 130.0
  FAR = 60.0
  """Degrees of tilt the way it pours must reach."""
  SIDE = 0.03
  """How far the hand is off the bowl's middle, pouring: the mouth swings that far back over it."""
  GRIP = props.mcup_tab_height()
  """Grasp height over the cup's foot: the pads on the two tabs."""
  OVER = 0.05
  """The cup's foot over the table, pouring: clear of the bowl's 3 cm rim, and low enough to reach near the base."""

  def __init__(self, arm: str, ik, rng, dt) -> None:
    home = np.array(b.MCUP_XY)
    self.poured = False

    def wanted(env, e):
      p = pos(env, "cup", e)
      standing = np.linalg.norm(p[:2] - home) < 0.06 and p[2] < b.Z + 0.01
      return [] if self.poured or not standing else ["cup"]

    def drop(env, e, name):
      # Its foot 1 cm over where it started, for the hand: let go and it stands.
      return np.r_[home, b.Z + 0.01 + self.GRIP - 0.02]

    def busy(other_site):
      return np.linalg.norm(other_site[:2] - home) < 0.14 and other_site[2] < b.Z + 0.30

    super().__init__(arm, ik, rng, dt, wanted, drop, busy, grasp_height=lambda name: self.GRIP, square=lambda name: True,
                     keep_jaw=True, opening=1.0, plumb_drop=True, approach=0.06)  # Plumb: held upright.

  def plan(self, env, e: int, now: float) -> None:
    if self.stage == "wait" and not self.poured and now >= self.start and not self.wanted(env, e):
      self.start = now + self.rng.uniform(*RETRY)  # The other arm has the cup: look again later.
      return
    if self.stage == "lifted" and not self.poured and self.holding(env, e):
      self._plan_over_bowl(env, e)
      return
    if self.stage == "pouring":
      self._plan_tilt(env, e, now)
      return
    super().plan(env, e, now)

  def _jaw(self, env, e: int, name: str):
    """Across the tabs, pointing forward for both arms: so held, the tilt that pours stays within joint 6's range,
    whichever arm it is.  The cup is the same either way round."""
    jaw = super()._jaw(env, e, name)
    return (abs(jaw[0]), jaw[1] if jaw[0] >= 0 else -jaw[1], 0.0)

  def out_of_cup(self, env, e: int) -> int:
    """How many beads have left the cup: in the air or in the bowl."""
    centre = pos(env, "cup", e) + [0.0, 0.0, 0.02]
    return sum(np.linalg.norm(pos(env, n, e) - centre) > 0.06 for n in b.BEADS)

  def act(self, env, e: int, now: float) -> np.ndarray:
    if self.stage == "pouring" and self.tilt < self.MAX_TILT and self.out_of_cup(env, e) - self.out_before >= self.TARGET:
      # Its share is on its way: tilt back at once, every step counts.  Those over the lip still land.
      self.queue.clear()
      self.q = self.current.copy()
      self.rot = self.upright
      self.glide(self.solve(self.pour_site), 0.5, 0.0)
      self.tilt = self.MAX_TILT
    return super().act(env, e, now)

  def holding(self, env, e: int) -> bool:
    """The cup's origin is its foot: measured from where it is held instead."""
    robot = env.scene[self.arm]
    fingers = float(robot.data.joint_pos[e, robot.joint_names.index("joint7")])
    q = env.scene["cup"].data.root_link_quat_w[e:e + 1]
    grip = pos(env, "cup", e) + quat_apply(q, q.new_tensor([[0.0, 0.0, self.GRIP]]))[0].cpu().numpy()  # Along its own axis.
    # The tabs are 8 cm across: closed on them the fingers are still most of the way open.
    return bool(np.linalg.norm(grip - site(env, self.arm, GRASP_SITE, e)) < 0.04 and fingers < FINGER_MAX - 0.003)

  def _plan_over_bowl(self, env, e: int) -> None:
    bowl = np.array(b.MBOWL_XY[self.arm])
    # How to hold it over the bowl, plumb, and which way to tilt: whichever reaches far enough to pour; most run out of
    # reach at once.  The cup is the same turned half round in the hand, so either way of holding it will do.  Then
    # hold the cup off to the side of the bowl so that, tilted, its mouth is over the bowl and its foot swings out past
    # the rim.
    found = None
    for jaw in (self.jaw, (-self.jaw[0], -self.jaw[1], 0.0)):
      upright = self.lean_rot(jaw, bowl, 0.0)
      across = (-jaw[1], jaw[0], 0.0)
      for sign in (1.0, -1.0):
        self.rot = turn(across, sign * self.FAR) @ upright
        mouth = (self.rot @ np.array([0.0, 0.0, -1.0]))[:2]  # The cup's up is the fingers' back.
        # The beads go over the lip on the side the cup leans to, and land a few centimetres beyond the hand.
        site = np.r_[bowl - self.SIDE * mouth / np.linalg.norm(mouth), b.Z + self.OVER + self.GRIP]
        try:
          self.solve(site)
        except RuntimeError:
          continue
        found = (jaw, upright, sign, site)
        break
      if found:
        break
    if found is None:
      jaw = self.jaw
      found = (jaw, self.lean_rot(jaw, bowl, 0.0), -1.0, np.r_[bowl, b.Z + self.OVER + self.GRIP])
    self.jaw, self.upright, self.sign, self.pour_site = found
    self.rot = self.upright
    self.glide(self.solve(self.pour_site + [0, 0, 0.02]), 2.0, 0.0)
    self.line(self.pour_site + [0, 0, 0.02], self.pour_site, 0.4, 0.0)
    self.tilt = 0.0
    self.out_before = self.out_of_cup(env, e)  # The other arm's share, poured before: not this pour's.
    self.stage = "pouring"

  def _plan_tilt(self, env, e: int, now: float) -> None:
    count = int(beads_in_bowl(env, self.arm)[e])
    if not self.holding(env, e):
      self.log.append(f"t={now:.1f}: dropped the cup")
      self.glide(READY_Q, 1.0, 1.0)
      self.stage, self.target, self.poured = "backoff", None, True
      return
    if count >= self.TARGET or self.tilt >= self.MAX_TILT:
      self.rot = self.upright  # Upright again, then the Picker stands the cup back.
      self.glide(self.solve(self.pour_site), 1.2, 0.0)
      self.hold(0.5, 0.0)
      self.log.append(f"t={now:.1f}: poured {count}")
      self.poured = True
      self.stage = "lifted"
      return
    # About the level line across the jaw: the pads turn with the cup.  About the jaw itself, the cup's weight twists it
    # in the pinch and it falls out.
    across = (-self.jaw[1], self.jaw[0], 0.0)
    step = self.STEP if self.tilt < self.SLOW_FROM else self.FINE  # Near where the beads begin to go: finely.
    self.rot = turn(across, self.sign * (self.tilt + step)) @ self.upright
    try:
      q = self.solve(self.pour_site)
    except RuntimeError:
      q = None
    if q is not None:
      self.tilt += step
      self.glide(q, 0.4, 0.0)
      self.hold(0.4 if step == self.STEP else 0.8, 0.0)
      return
    self.log.append(f"t={now:.1f}: tilt reaches only {self.tilt:.0f} deg")
    self.tilt = self.MAX_TILT  # As far as it reaches: done.
