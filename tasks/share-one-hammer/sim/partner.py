"""The second arm's script, and the script the demonstration runs: an arm that waits its turn for the one hammer,
drives its own nail and puts the hammer back.  It reads the scene's true state.

    arm = Sharer(arm, ik, rng, dt)
    row = arm.act(env, e, now)       # every step: 6 joint targets (rad) and the gripper, 0 closed to 1 open

1. Wait a random while, then go for the hammer only if it lies still near where it started, nobody holds it and the
   other gripper is not coming for it; otherwise look again after a random while.  If, on the way, the other gripper
   comes for it too, back off and wait a random while.
2. Pick it up by the middle of the handle, fingers plumb along it; lift, lean, and take it over its own nail.
3. Strike straight down, each blow aimed from where the hammer's face is now, until the nail is home; if the hammer
   has slid in the hand out of reach, set it down and take a fresh grip.
4. Set the hammer down where it started, as it sits in the hand, let go and go back to ready."""

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
from mjlab.utils.lab_api.math import quat_apply_inverse

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


LEAN = 8.0
"""Degrees the fingertips lean forward, about the jaw, over the nail: joint 4's limits need some."""
LIFT = 0.045
THROUGH = 0.03
STROKE = 0.12
CYCLE = 0.30
MAX_BLOWS = 300
FIRST_TRY = (0.0, 6.0)
RETRY = (0.5, 2.5)
CLEAR = 0.05
"""The other gripper is coming for the hammer when it is this much nearer the handle than it is at ready."""
HOME = props.NAIL_STANDOUT - 0.008


def grip_rotation(lean: float, yaw: float = 0.0) -> np.ndarray:
  """Fingers down, the jaw across a handle that points ``yaw`` radians from +x, leaning ``lean`` degrees forward."""
  c, s = np.cos(yaw), np.sin(yaw)
  down = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.column_stack([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
  return turn((0.0, -1.0, 0.0), lean) @ down


def holding(env: ManagerBasedRlEnv, arm: str, e: int) -> bool:
  """The arm has the hammer: its grasp site within 5 cm of the handle and its fingers closing."""
  hammer = env.scene["hammer"].data
  here = env.scene[arm].data.site_pos_w[e:e + 1, env.scene[arm].site_names.index(GRASP_SITE)]
  local = quat_apply_inverse(hammer.root_link_quat_w[e:e + 1], here - hammer.root_link_pos_w[e:e + 1])[0]
  off_axis = float(local[1:].norm() + (local[0].abs() - props.HAMMER_HANDLE / 2).clamp(min=0.0))
  robot = env.scene[arm]
  fingers = float(robot.data.joint_pos[e, robot.joint_names.index("joint7")])
  return off_axis < 0.05 and fingers < 0.8 * FINGER_MAX


def nail_depth(env: ManagerBasedRlEnv, arm: str, e: int) -> float:
  return float(-env.scene[b.NAILS[arm]].data.joint_pos[e, 0]) + 0.0  # + 0.0: never -0.0


class Sharer(Mover):
  def __init__(self, arm, ik, rng, dt) -> None:
    super().__init__(arm, ik, rng, dt)
    self.other = [a for a in b.ARMS if a != arm][0]
    self.start = rng.uniform(*FIRST_TRY)
    self.blows = self.backoffs = 0
    self.home = None
    self.got_hammer_at = self.home_at = self.returned_at = None

  def _first_look(self, env: ManagerBasedRlEnv, e: int) -> None:
    """At the start, both arms at ready: where the hammer is, and how far the other gripper is from its handle."""
    self.home = pos(env, "hammer", e)
    self.handle = self.home.copy()
    self.other_at_ready = np.linalg.norm(site(env, self.other, GRASP_SITE, e) - self.handle)

  def _coming(self, env: ManagerBasedRlEnv, e: int) -> bool:
    return np.linalg.norm(site(env, self.other, GRASP_SITE, e) - self.handle) < self.other_at_ready - CLEAR

  def act(self, env: ManagerBasedRlEnv, e: int, now: float) -> np.ndarray:
    if self.home is None:
      self._first_look(env, e)
    if self.stage == "approach" and self.queue and self._coming(env, e):
      # The other arm is going for the hammer as well: back off from where the arm is now, and try again later.
      self.queue.clear()
      self.q = self.current.copy()
      self.glide(READY_Q, 0.8, 1.0)
      self.stage, self.backoffs = "backoff", self.backoffs + 1
      self.log.append(f"t={now:.1f}: the other arm is coming for the hammer too; backing off")
    return super().act(env, e, now)

  def wait_again(self, now: float) -> None:
    self.start = now + self.rng.uniform(*RETRY)
    self.stage = "wait"

  def put_back(self, then: str) -> None:
    self.rot = self.pick
    self.glide(self.solve(self.grasp + [0, 0, 0.07]), 1.5, 0.0)
    self.stage, self.after = "placing", then

  def plan(self, env: ManagerBasedRlEnv, e: int, now: float) -> None:
    if self.stage == "wait":
      if now < self.start:
        return
      hammer_pos = pos(env, "hammer", e)
      still = float(env.scene["hammer"].data.root_link_lin_vel_w[e].norm()) < 0.05
      free = (np.linalg.norm(hammer_pos[:2] - self.home[:2]) < 0.25 and hammer_pos[2] < b.Z + 0.05 and still
              and not holding(env, self.arm, e) and not holding(env, self.other, e))
      if not free or self._coming(env, e):
        self.wait_again(now)
        return
      q = env.scene["hammer"].data.root_link_quat_w[e:e + 1]
      x = quat_apply(q, q.new_tensor([[1.0, 0.0, 0.0]]))[0].cpu().numpy()
      yaw = float(np.arctan2(x[1], x[0]))
      self.rot = self.pick = grip_rotation(0.0, yaw)
      self.lean = grip_rotation(LEAN, yaw)
      self.grasp = hammer_pos.copy()
      self.grasp[2] = max(float(hammer_pos[2]), b.Z + 0.02) - 0.01  # Mid-pad: 1 cm below the handle's axis.
      self.glide(self.solve(self.grasp + [0, 0, 0.08]), 1.5, 1.0)
      self.stage = "approach"
      self.log.append(f"t={now:.1f}: the hammer is free; going for it")
    elif self.stage == "backoff":
      self.wait_again(now)
    elif self.stage == "approach":
      self.line(self.grasp + [0, 0, 0.08], self.grasp, 0.8, 1.0)
      self.hold(1.0, 0.0)
      self.line(self.grasp, self.grasp + [0, 0, 0.08], 0.8, 0.0)
      self.rot = self.lean
      self.glide(self.solve(self.grasp + [0, 0, 0.08]), 0.8, 0.0)
      self.hold(0.3, 0.0)
      self.stage = "lifted"
    elif self.stage == "lifted":
      here = site(env, self.arm, GRASP_SITE, e)
      self.offset = site(env, "hammer", "face", e) - here
      if not holding(env, self.arm, e) or self.offset[2] < -0.05:
        self.log.append(f"t={now:.1f}: the hammer is not held level; putting it back to try again")
        self.put_back("regrip")
        return
      self.got_hammer_at = now
      head = site(env, b.NAILS[self.arm], "head", e)
      self.glide(self.solve(head + [0, 0, LIFT] - self.offset), 1.5, 0.0)
      self.stage = "strike"
    elif self.stage == "strike":
      here = site(env, self.arm, GRASP_SITE, e)
      face = site(env, "hammer", "face", e)
      if np.linalg.norm(face - here - self.offset) < 0.03:
        self.offset = face - here
      head = site(env, b.NAILS[self.arm], "head", e)
      top = head + [0, 0, LIFT] - self.offset
      if nail_depth(env, self.arm, e) >= HOME or self.blows >= MAX_BLOWS:
        if nail_depth(env, self.arm, e) >= HOME:
          self.home_at = now
        self.log.append(f"t={now:.1f}: {'home' if self.home_at else 'out of blows'} after {self.blows} blows")
        self.line(top, top + [0, 0, 0.04], 0.4, 0.0)
        self.put_back("returning")
        return
      bottom = head + [0, 0, -THROUGH] - self.offset
      queued, q = len(self.queue), self.q.copy()
      try:
        self.line(top, bottom, STROKE, 0.0, through_table=True)
        self.hold(CYCLE - STROKE, 0.0)
        self.line(bottom, top, STROKE, 0.0, through_table=True)
        self.hold(CYCLE - STROKE, 0.0)
      except RuntimeError:
        # The hammer has slid or turned in the fingers, and the blow is out of this hand's reach: set it down and take a
        # fresh grip, then carry on.
        del self.queue[queued:]
        self.q = q
        self.log.append(f"t={now:.1f}: the hammer has slid in the hand; putting it down to take a fresh grip")
        self.put_back("regrip")
        return
      self.blows += 1
    elif self.stage == "placing":
      # The hammer slides along its handle in the fingers as it strikes: aim the hand so that the hammer, as it sits in
      # it now, lands where it started.
      here = site(env, self.arm, GRASP_SITE, e)
      in_hand = pos(env, "hammer", e) - here
      target = self.home + [0, 0, 0.004] - in_hand
      target[2] = max(target[2], b.Z + 0.025)
      try:
        self.solve(target + [0, 0, 0.06])
        self.solve(target)
      except RuntimeError:
        # Turned in the hand, its place is out of this hand's reach: lower it straight down where it is instead.
        target = here.copy()
        target[2] = max(self.home[2] + 0.004 - in_hand[2], b.Z + 0.025)
        self.log.append(f"t={now:.1f}: cannot reach the hammer's place; setting it down where it is")
      self.line(here, target + [0, 0, 0.06], 0.8, 0.0)
      self.line(target + [0, 0, 0.06], target, 0.8, 0.0)
      self.hold(0.6, 1.0)
      self.line(target, target + [0, 0, 0.08], 0.6, 1.0)
      self.glide(READY_Q, 1.2, 1.0)
      self.stage = self.after
    elif self.stage == "regrip":
      self.wait_again(now)
    elif self.stage == "returning":
      self.returned_at = now
      self.log.append(f"t={now:.1f}: put the hammer back; done")
      self.stage = "done"
