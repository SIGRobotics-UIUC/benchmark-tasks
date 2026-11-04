"""The second arm's script, and the script the demonstration was recorded from: one arm holds the nail, the other
hammers it.  Each reads the scene's true state.

    arm = Holder(arm, ik, rng, dt)        # or Hammerer(...)
    row = arm.act(env, e, now)            # every step: 6 joint targets (rad) and the gripper, 0 closed to 1 open

The holder takes the nail off its rack, turns it upright and stands its point in the hole, letting it find its way in
loosely; then it grips it firmly and presses it in, following it down, while the hammer taps; at ``SET`` deep the nail
stands by itself, and the holder lets go, draws back along its fingers and rises clear.  The hammerer waits until the
nail stands in the hole, takes the hammer, taps while the nail is held, waits for the holding hand to be ``CLEAR`` away,
then drives the nail home with full blows and puts the hammer back."""

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

START = (0.0, 4.0)
PRESS = 0.006
LOOSE = 0.0002
"""How much wider than its squeeze the holder holds the nail while it lets the point find its way into the hole."""
CLEAR = 0.12
"""The hammerer strikes only while the other hand holds still, or is at least this far from the nail's head."""
SET = 0.012
"""The holder lets go once the nail is this deep: it then stands by itself."""
HOME = props.SPIKE_DRIVE
LEAN = 30.0
LIFT = 0.03
THROUGH = (0.01, 0.03)
"""How far below the nail's head a blow is aimed: short while the nail is not yet set, as it may be held; then full."""
STROKE = 0.12
"""Seconds for a full blow to come down (and to go back up)."""
CYCLE = 0.30
"""Seconds from one blow to the next, at the top."""
TAP = (0.015, 0.20)
FIRM = SET
"""Light taps until the nail is this deep: while it is held."""
MAX_BLOWS = 300
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


def depth(env: ManagerBasedRlEnv, e: int) -> float:
  return float(-env.scene["post"].data.joint_pos[e, 0].item()) + 0.0  # + 0.0: never -0.0


def nail_axis(env: ManagerBasedRlEnv, e: int) -> np.ndarray:
  q = env.scene["spike"].data.root_link_quat_w[e:e + 1]
  return quat_apply(q, q.new_tensor([[0.0, 0.0, 1.0]]))[0].cpu().numpy()


def nail_standing(env: ManagerBasedRlEnv, e: int) -> bool:
  """The nail's point is in the hole and it is upright, held or not: within 8 degrees until it is set, within 12 once it
  is (set, it leans on the side of the hole, and blows straighten it as they drive it)."""
  tip = site(env, "spike", "tip", e)
  hole = site(env, "post", "hole", e)
  limit = 12 if depth(env, e) >= SET else 8
  return bool(np.linalg.norm(tip[:2] - hole[:2]) < 0.004 and tip[2] < hole[2] and nail_axis(env, e)[2] > np.cos(np.radians(limit)))


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


class Holder(Mover):
  def __init__(self, arm: str, ik: ArmIk, rng: np.random.Generator, dt: float) -> None:
    super().__init__(arm, ik, rng, dt)
    # The nail lies along y: the jaw closes across it, along x, turned to suit each arm (the gripper is the same either
    # way round, and the wrist cannot turn the half turn between them).
    self.pick = plumb((self.side, 0.0, 0.0))
    # Turned a quarter turn about the jaw, fingers level and pointing at the middle of the table: the nail upright.
    self.upright = turn((1.0, 0.0, 0.0), -90.0 * self.side) @ self.pick
    self.in_hand = None

  def plan(self, env: ManagerBasedRlEnv, e: int, now: float) -> None:
    if self.stage == "wait":
      if now < self.start:
        return
      nail = pos(env, "spike", e)
      self.rot = self.pick
      # Near the point, which sticks out past the rack; the shank's axis at the grasp site (the pads reach 1 cm below).
      axis = nail_axis(env, e)
      self.grasp = nail - props.SPIKE_GRIP * axis
      self.glide(self.solve(self.grasp + [0, 0, 0.07]), 1.5, 1.0)
      self.line(self.grasp + [0, 0, 0.07], self.grasp, 0.8, 1.0)
      self.hold(0.8, 0.0)
      self.line(self.grasp, self.grasp + [0, 0, 0.08], 0.8, 0.0)
      self.stage = "turn"
    elif self.stage == "turn":
      # Upright, high over the hole: fingers level, this hand reaches only close over the post.
      self.rot = self.upright
      over = np.array([b.POST_XY[0], b.POST_XY[1], b.Z + 0.18])
      self.glide(self.solve(over), 1.5, 0.0)
      self.hold(0.3, 0.0)
      self.stage = "place"
    elif self.stage == "place":
      grasp_site = site(env, self.arm, GRASP_SITE, e)
      self.in_hand = pos(env, "spike", e) - grasp_site
      if nail_axis(env, e)[2] < np.cos(np.radians(15)):
        raise RuntimeError("the nail is not upright in the hand")
      # The point just into the socket, the nail's middle over the hole.
      hole = site(env, "post", "hole", e)
      centre = np.array([hole[0], hole[1], hole[2] - props.POST_SOCKET + props.SPIKE_SHANK[1] + 0.001])
      self.target = centre - self.in_hand
      # Loosely down into the hole, so that the point can find its own way in (the hole is only 0.6 mm wider than the
      # point each side, finer than this hand places it); once in, a firm grip for the taps.
      robot = env.scene[self.arm]
      closed = float(robot.data.joint_pos[e, robot.joint_names.index("joint7")].item())
      loose = (closed + LOOSE) / FINGER_MAX
      self.line(grasp_site, self.target + [0, 0, 0.03], 1.0, 0.0)
      self.hold(0.2, loose)
      self.line(self.target + [0, 0, 0.03], self.target, 0.8, loose)
      self.hold(0.5, loose)
      self.hold(0.3, 0.0)
      self.stage = "hold"
      self.log.append(f"t={now:.1f}: the nail's point is in the hole; holding it firmly")
    elif self.stage == "hold":
      if depth(env, e) >= SET:
        # Set: let go, draw the open fingers back off the nail along themselves, then lift clear and go to ready.  A
        # joint-space glide straight to ready swung the open fingers sideways through the nail and threw it out.
        here = site(env, self.arm, GRASP_SITE, e)
        self.hold(0.6, 1.0)
        # As far back as the hand reaches with its fingers level (about 2 cm from over the post).
        for retreat in (0.02, 0.016, 0.012):
          back = here - retreat * self.rot[:, 2]
          try:
            self.solve(back)
          except RuntimeError:
            continue
          self.line(here, back, 0.8, 1.0)
          # Then straight up behind the nail, above its head, before swinging away: the fingertips are still close.
          self.line(back, back + [0, 0, 0.09], 1.0, 1.0)
          break
        self.glide(READY_Q, 2.0, 1.0)
        self.stage = "release"
        self.log.append(f"t={now:.1f}: the nail is set; letting go")
      else:
        # A firm grip keeps it upright; the hand goes down with it as the taps drive it, pressing a little ahead, so
        # it is never driven out through the fingers and never tips in them.
        self.q = self.solve(self.target - [0, 0, depth(env, e) + PRESS])
        self.hold(0.1, 0.0)
    elif self.stage == "release":
      self.stage = "done"


class Hammerer(Mover):
  def __init__(self, arm: str, ik: ArmIk, rng: np.random.Generator, dt: float) -> None:
    super().__init__(arm, ik, rng, dt)
    # The handle lies along y, head toward the middle: the jaw across it, along x, turned to suit each arm; over the
    # nail the fingers lean toward the middle of the table, about the jaw.
    # Picked up plumb, the pads along the handle; leant toward the middle of the table, about the jaw, once lifted.
    self.pick = plumb((self.side, 0.0, 0.0))
    self.lean = turn((1.0, 0.0, 0.0), -self.side * LEAN) @ self.pick
    self.blows = 0
    self.offset = None

  def plan(self, env: ManagerBasedRlEnv, e: int, now: float) -> None:
    if self.stage == "wait":
      if now < self.start:
        return
      if not nail_standing(env, e):
        if int(now / self.dt) % int(5.0 / self.dt) == 0:  # Every 5 s, say why.
          tip, hole = site(env, "spike", "tip", e), site(env, "post", "hole", e)
          self.log.append(f"t={now:.1f}: waiting for the nail to stand: point {1000 * np.linalg.norm(tip[:2] - hole[:2]):.1f} mm "
                          f"off the hole, {1000 * (tip[2] - hole[2]):.1f} mm from its top, leaning "
                          f"{np.degrees(np.arccos(np.clip(nail_axis(env, e)[2], -1, 1))):.1f} deg")
        return
      self.home = pos(env, "hammer", e)
      self.rot = self.pick
      self.grasp = self.home + [0.0, 0.0, -0.01]  # Mid-pad: 1 cm below the handle's axis.
      self.glide(self.solve(self.grasp + [0, 0, 0.08]), 1.5, 1.0)
      self.line(self.grasp + [0, 0, 0.08], self.grasp, 0.8, 1.0)
      self.hold(0.8, 0.0)
      self.line(self.grasp, self.grasp + [0, 0, 0.08], 0.8, 0.0)
      self.rot = self.lean
      self.glide(self.solve(self.grasp + [0, 0, 0.08]), 0.8, 0.0)
      self.hold(0.3, 0.0)
      self.stage = "lifted"
      self.log.append(f"t={now:.1f}: the nail stands in the hole; taking the hammer")
    elif self.stage == "lifted":
      here = site(env, self.arm, GRASP_SITE, e)
      self.offset = site(env, "hammer", "face", e) - here
      self.log.append(f"t={now:.1f}: lifted; face {np.round(1000 * self.offset).astype(int).tolist()} mm from the hand")
      if self.offset[2] < -0.07:
        raise RuntimeError("the hammer turned in the grip")
      head = site(env, "spike", "head", e)
      top = head + [0, 0, TAP[0]] - self.offset
      # In straight lines (a joint-space glide swung the hammer through the standing nail), first to a point over its
      # own side of the nail and above, then across and down: straight from the hammer's rest, it crossed the holding
      # hand.
      high = top + [0, self.side * 0.03, 0.015]
      self.line(here, high, 1.5, 0.0)
      self.line(high, top, 0.6, 0.0)
      self.stage = "strike"
    elif self.stage == "strike":
      here = site(env, self.arm, GRASP_SITE, e)
      face = site(env, "hammer", "face", e)
      if self.blows in (1, 5) and getattr(self, "logged_blow", -1) != self.blows:
        self.logged_blow = self.blows
        self.log.append(f"t={now:.1f}: blow {self.blows}; face {np.round(1000 * (face - here)).astype(int).tolist()} mm from the hand")
      if np.linalg.norm(face - here - self.offset) < 0.03:
        self.offset = face - here
      head = site(env, "spike", "head", e)
      setting = depth(env, e) < FIRM
      top = head + [0, 0, TAP[0] if setting else LIFT] - self.offset
      other_arm = [a for a in b.ARMS if a != self.arm][0]
      moving = float(env.scene[other_arm].data.joint_vel[e, :6].abs().max().item()) > 0.15
      near = np.linalg.norm(site(env, other_arm, GRASP_SITE, e) - head) < CLEAR
      reason = None
      if near and depth(env, e) >= SET and depth(env, e) < HOME:
        reason = "the nail is set, and the other hand is still near it"
      elif moving and near and depth(env, e) < HOME:
        reason = "the other hand is moving near the nail"
      elif not nail_standing(env, e) and depth(env, e) < HOME:
        tip, hole = site(env, "spike", "tip", e), site(env, "post", "hole", e)
        reason = (f"the nail is not standing (point {1000 * np.linalg.norm(tip[:2] - hole[:2]):.1f} mm off the hole, "
                  f"{1000 * (tip[2] - hole[2]):.1f} mm from its top; leaning {np.degrees(np.arccos(nail_axis(env, e)[2])):.1f} deg)")
      if reason:
        self.hold(0.2, 0.0)  # Wait at the top of the stroke, and look again.
        if reason != getattr(self, "last_reason", None):
          self.log.append(f"t={now:.1f}: waiting: {reason}")
          self.last_reason = reason
        return
      self.last_reason = None
      if depth(env, e) >= HOME or self.blows >= MAX_BLOWS:
        self.log.append(f"t={now:.1f}: {'home' if depth(env, e) >= HOME else 'out of blows'} after {self.blows} blows")
        self.rot = self.pick
        self.line(top, top + [0, 0, 0.03], 0.4, 0.0)
        self.glide(self.solve(self.grasp + [0, 0, 0.07]), 1.5, 0.0)
        self.stage = "put_back"
        return
      bottom = head + [0, 0, -THROUGH[not setting]] - self.offset
      stroke = TAP[1] if setting else STROKE
      self.line(top, bottom, stroke, 0.0, through_table=True)
      # A light tap comes straight back up (held down on the nail, the leaning hammer was pushed back along its handle
      # through the grip); a full blow follows through, pressing at the bottom, which is what drives the nail.
      dwell = 0.0 if setting else CYCLE - STROKE
      if dwell:
        self.hold(dwell, 0.0)
      self.line(bottom, top, stroke, 0.0, through_table=True)
      self.hold(2 * (CYCLE - STROKE) - dwell, 0.0)
      self.blows += 0 if setting else 1  # Light taps while it is held do not count against the limit.
    elif self.stage == "put_back":
      # Set it down where it started, as it now sits in the hand.
      here = site(env, self.arm, GRASP_SITE, e)
      target = self.home + [0, 0, 0.004] - (pos(env, "hammer", e) - here)
      target[2] = max(target[2], b.Z + 0.025)
      self.line(here, target + [0, 0, 0.05], 0.8, 0.0)
      self.line(target + [0, 0, 0.05], target, 0.8, 0.0)
      self.hold(0.6, 1.0)
      self.line(target, target + [0, 0, 0.08], 0.6, 1.0)
      self.glide(READY_Q, 1.2, 1.0)
      self.stage = "release"
    elif self.stage == "release":
      self.stage = "done"
