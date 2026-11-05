"""The scripted arms, each reading the scene's true state.

    arm = Picker(arm, ik, rng, dt)        # Arm 1: picks the red blocks into its bin (the task's other arm)
    arm = Sweeper(arm, ik, rng, dt)       # the demonstration: sweeps the clutter off the table, yielding to Arm 1
    row = arm.act(env, e, now)            # every step: 6 joint targets (rad) and the gripper, 0 closed to 1 open

The picker waits ``START`` seconds, then takes the red blocks to its bin one at a time, in a fixed order and at the same
pace every trial, skipping any block that is no longer on the table.  The sweeper picks up the broom and makes one
stroke toward its own side edge through each band of the table, starting each only when the two arms will stay apart,
sweeps again at any clutter left, and stands the broom back up."""

from __future__ import annotations

import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply

from sim import benchmark as b
from sim import props
from sim.env_cfg import READY_JOINTS
from sim.ik import ArmIk
from sim.robot import GRASP_SITE
from sim.scene import BASE_POSE, LEFT_ARM, TABLE_CENTER_X, TABLE_SIZE, world_to_base

START = (2.0, 2.0)
"""Seconds the picker waits before its first block (a range, for a seeded random start; fixed here)."""
READY_Q = np.array(list(READY_JOINTS.values()))
LEAN = props.HANDLE_TILT
"""Both arms hold their hands leaning forward by this much, about the jaw: fingers straight down, the YAM reaches only
the near half of the table."""


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


def on_table(p: np.ndarray) -> bool:
  """A point is over the table top and not fallen below it.  A position that is not finite (the simulation blew up)
  counts as on the table: never as cleared."""
  if not np.isfinite(p).all():
    return True
  half_x, half_y = TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2
  return bool(abs(p[0] - TABLE_CENTER_X) < half_x and abs(p[1]) < half_y and p[2] > b.Z - 0.02)


def in_bin(p: np.ndarray) -> bool:
  if not np.isfinite(p).all():
    return False
  ix, iy, h, _ = props.BIN
  return bool(abs(p[0] - b.BIN_XY[0]) < ix / 2 and abs(p[1] - b.BIN_XY[1]) < iy / 2 and p[2] < b.Z + h + 0.03)


def block_yaw(env: ManagerBasedRlEnv, name: str, e: int) -> float:
  q = env.scene[name].data.root_link_quat_w[e:e + 1]
  x = quat_apply(q, q.new_tensor([[1.0, 0.0, 0.0]]))[0].cpu().numpy()
  return float(np.arctan2(x[1], x[0]))


def leaned(jaw: tuple[float, float, float]) -> np.ndarray:
  """Fingers leaning ``LEAN`` forward of straight down, about the jaw (closing direction) along the horizontal ``jaw``."""
  return turn(jaw, -LEAN) @ plumb(jaw)


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


class Picker(Mover):
  """Arm 1: the red blocks, one by one, into the bin."""

  CARRY = 0.15
  """Height above the table at which a block is carried and dropped over the bin."""

  def __init__(self, arm: str, ik: ArmIk, rng: np.random.Generator, dt: float) -> None:
    super().__init__(arm, ik, rng, dt)
    self.todo = list(b.BLOCK_NAMES)
    self.holding: str | None = None

  def plan(self, env: ManagerBasedRlEnv, e: int, now: float) -> None:
    if self.stage == "wait":
      if now >= self.start:
        self.stage = "next"
      return
    if self.stage == "next":
      self.holding = None
      while self.todo:
        name = self.todo.pop(0)
        p = pos(env, name, e)
        if on_table(p) and not in_bin(p):
          break
        self.log.append(f"t={now:.1f}: {name} is gone; skipping it")
      else:
        # Done: wait high over the bin, out of the way.
        self.rot = leaned((0.0, 1.0, 0.0))
        self.glide(self.solve([b.BIN_XY[0], b.BIN_XY[1], b.Z + 0.20]), 1.5, 1.0)
        self.stage = "release"
        return
      # The jaw across the block's faces, whichever pair is nearest across the table (y).
      yaw = block_yaw(env, name, e)
      yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
      self.rot = leaned((-np.sin(yaw), np.cos(yaw), 0.0))
      grasp = p + [0.0, 0.0, 0.004]
      for rise in (0.10, 0.08, 0.06):  # As high above the block as the arm reaches there.
        above = grasp + [0, 0, rise]
        try:
          q_above = self.solve(above)
          self.solve(grasp)
        except RuntimeError:
          continue
        break
      else:
        self.log.append(f"t={now:.1f}: cannot reach {name} at {np.round(p, 3).tolist()}; skipping it")
        self.holding = None
        return
      self.glide(q_above, 1.8, 1.0)
      self.line(above, grasp, 0.8, 1.0)
      self.hold(0.5, 0.0)
      self.line(grasp, above, 0.8, 0.0)
      self.holding = name
      self.stage = "carry"
      self.log.append(f"t={now:.1f}: taking {name}")
    elif self.stage == "carry":
      self.rot = leaned((0.0, 1.0, 0.0))
      # Each block to its own spot in the bin, so they don't stack; the bin's middle if the arm can't reach the spot.
      spot = [(0.04, 0.035), (0.04, -0.035), (0.0, 0.035), (0.0, -0.035)][b.BLOCK_NAMES.index(self.holding) % 4]
      try:
        q = self.solve([b.BIN_XY[0] + spot[0], b.BIN_XY[1] + spot[1], b.Z + self.CARRY])
      except RuntimeError:
        q = self.solve([b.BIN_XY[0], b.BIN_XY[1], b.Z + self.CARRY])
      self.glide(q, 1.8, 0.0)
      self.hold(0.5, 1.0)
      self.stage = "dropped"
    elif self.stage == "dropped":
      self.holding = None
      self.stage = "next"
    elif self.stage == "release":
      self.stage = "done"


class Sweeper(Mover):
  """The demonstration: picks up the broom and sweeps toward its own side edge, one band at a time, starting a stroke
  only when the two arms will stay ``KEEP_OUT`` apart throughout it; sweeps again at any clutter left; then stands the
  broom back up.  It leaves the red blocks to Arm 1."""

  BANDS = ((0.17, 0.08), (0.30, 0.08), (0.42, 0.08))
  """(grip x, start y) of the strokes, one per band, over this arm's half of the table.  None nearer this arm's base
  than x = 0.17: the broom caught on it."""
  CLEARANCE = 0.010
  """The bristles' tips ride this far above the table: every item is at least 3 cm tall, and bristles dragging on the
  table caught at its edge and twisted the broom out of the hand."""
  KEEP_OUT = 0.09
  """A stroke starts only if every point of this arm stays this far from every point of Arm 1 (link shells are about
  4 cm thick)."""
  EDGE = TABLE_SIZE[1] / 2 + 0.02
  """Strokes end with the broom's middle this far out: its front face 4.5 cm past the edge."""
  PACE = 3.0
  """Seconds per metre of stroke."""
  LIFT = 0.04
  """How high the bristles come down onto the table from, at the start of a stroke, and go up at its end."""

  def __init__(self, arm: str, ik: ArmIk, rng: np.random.Generator, dt: float) -> None:
    super().__init__(arm, ik, rng, dt)
    self.start = 0.0
    self.rot = leaned((0.0, 1.0, 0.0))  # The jaw across the grip, the broom's length along x.
    self.todo = list(self.BANDS)
    self.extra = 0
    self.picker: Picker | None = None
    self.next_q: dict[str, np.ndarray | None] = {}

  def _points(self, q: np.ndarray, arm: str) -> np.ndarray:
    """Points along an arm at joints ``q``: each link's origin, three points between each pair, and the grasp point."""
    self.ik._set(q)
    data, model = self.ik.data, self.ik.model
    chain = [data.xpos[model.body(n).id] for n in ("link2", "link3", "link4", "link5", "link6", "gripper")]
    chain.append(data.site_xpos[self.ik.site])
    chain += [u + (v - u) * f for u, v in zip(chain[:-1], chain[1:]) for f in (0.25, 0.5, 0.75)]
    return np.array(chain) + np.array(BASE_POSE[arm][0])

  def _arm1_configs(self, env: ManagerBasedRlEnv, e: int, seconds: float) -> list[np.ndarray]:
    """Arm 1's joints now and over the next ``seconds``: its routine is fixed, so its queued joint targets say where it
    is going; once they run out, it glides to above the next red block it will take."""
    other = [a for a in b.ARMS if a != self.arm][0]
    now = env.scene[other].data.joint_pos[e, :6].cpu().numpy()
    picker = self.picker
    if picker is None:
      return [now]
    n = int(seconds / picker.dt)
    configs = [now] + [q for q, _ in picker.queue[:n:5]]
    if len(picker.queue) < n and picker.stage in ("wait", "next", "carry", "dropped"):
      for name in [m for m in picker.todo if on_table(pos(env, m, e))][:1]:
        if name not in self.next_q:
          picker_rot, picker.rot = picker.rot, leaned((0.0, 1.0, 0.0))
          try:
            self.next_q[name] = picker.solve(pos(env, name, e) + [0, 0, 0.10])
          except RuntimeError:
            self.next_q[name] = None
          picker.rot = picker_rot
        if self.next_q[name] is not None:
          last = configs[-1]
          configs += [last + (self.next_q[name] - last) * s for s in np.linspace(0.1, 1.0, 8)]
    return configs

  def _clear_of_arm1(self, env: ManagerBasedRlEnv, e: int, start: np.ndarray) -> bool:
    """Whether a stroke from ``start`` keeps the whole of this arm ``KEEP_OUT`` from the whole of Arm 1, as both move."""
    other = [a for a in b.ARMS if a != self.arm][0]
    end = np.array([start[0], self.EDGE, start[2]])
    mine = [self.q] + [self.solve(p) for p in (start + [0, 0, self.LIFT], start, 0.5 * (start + end), end)]
    ours = np.concatenate([self._points(q, self.arm) for q in mine])
    seconds = 2.2 + self.PACE * (self.EDGE - start[1])
    theirs = np.concatenate([self._points(q, other) for q in self._arm1_configs(env, e, seconds)])
    gap = np.linalg.norm(ours[:, None] - theirs[None], axis=-1).min()
    return gap >= self.KEEP_OUT

  def _start(self, x: float, y0: float) -> np.ndarray:
    low = b.Z + props.GRIP_HEIGHT + self.CLEARANCE
    for y in np.arange(y0, 0.36, 0.02):
      start = np.array([x, y, low])
      try:
        self.solve(start + [0, 0, self.LIFT])
        self.solve(start)
      except RuntimeError:
        continue
      return start
    raise RuntimeError(f"no reachable start at x={x:.2f}")

  def _stroke(self, start: np.ndarray) -> None:
    end = np.array([start[0], self.EDGE, start[2]])
    self.glide(self.solve(start + [0, 0, self.LIFT]), 1.2, 0.0)
    self.line(start + [0, 0, self.LIFT], start, 0.5, 0.0)
    self.line(start, end, self.PACE * (self.EDGE - start[1]), 0.0)
    self.line(end, end + [0, 0, self.LIFT], 0.4, 0.0)

  def _park(self) -> None:
    """Out of Arm 1's way: up, and over its own side's edge."""
    park = np.array([0.22, 0.40, b.Z + props.GRIP_HEIGHT + 0.08])
    self.glide(self.solve(park), 1.0, 0.0)

  def _grab(self, env: ManagerBasedRlEnv, e: int) -> None:
    grip = site(env, "broom", "grip", e)
    above = grip + [0, 0, 0.08]
    self.glide(self.solve(above), 1.5, 1.0)
    self.line(above, grip, 0.8, 1.0)
    self.hold(0.6, 0.0)
    self.line(grip, grip + [0, 0, 0.08], 0.8, 0.0)

  def _put_back(self) -> None:
    """Stand the broom back where it started, let go, and go to the ready pose."""
    spot = np.array([*b.BROOM_XY, b.Z + props.GRIP_HEIGHT + 0.004])
    self.glide(self.solve(spot + [0, 0, 0.06]), 1.5, 0.0)
    self.line(spot + [0, 0, 0.06], spot, 0.8, 0.0)
    self.hold(0.6, 1.0)
    self.line(spot, spot + [0, 0, 0.08], 0.8, 1.0)
    self.glide(READY_Q, 1.5, 1.0)

  def _left(self, env: ManagerBasedRlEnv, e: int) -> list[np.ndarray]:
    """Clutter still on the table."""
    return [pos(env, n, e) for n in b.CLUTTER_NAMES if on_table(pos(env, n, e))]

  def plan(self, env: ManagerBasedRlEnv, e: int, now: float) -> None:
    if self.stage == "wait":
      if now < 0.5:  # Until the simulation has stepped, the broom's pose is not yet read back.
        return
      self._grab(env, e)
      self._park()
      self.stage = "sweep"
      return
    if self.stage != "sweep":
      return
    # Lost the broom (it is more than 3 cm from the hand)?  Put it down where it is and pick it up again.
    if np.linalg.norm(site(env, "broom", "grip", e) - site(env, self.arm, GRASP_SITE, e)) > 0.03:
      self.slips = getattr(self, "slips", 0) + 1
      if self.slips > 2 or not on_table(pos(env, "broom", e)):
        self.log.append(f"t={now:.1f}: the broom is lost; stopping")
        self.stage = "done"
        self.hold(0.5, 1.0)
        return
      self.log.append(f"t={now:.1f}: the broom slipped; picking it up again")
      self.rot = leaned((0.0, 1.0, 0.0))
      self.hold(0.5, 1.0)
      self._grab(env, e)
      self._park()
      return
    if not self.todo:
      left = self._left(env, e)
      if not left or self.extra >= 12:
        self._put_back()
        self.stage = "done"
        return
      p = min(left, key=lambda p: p[1])  # The item farthest from the edge: sweeping it takes others with it.
      self.todo.append((float(np.clip(p[0], 0.17, 0.42)), max(p[1] - 0.07, -0.12)))
      self.extra += 1
    # The first stroke Arm 1 is clear of; if none, wait parked and look again.
    for i, (x, y0) in enumerate(self.todo):
      try:
        start = self._start(x, y0)
      except RuntimeError as err:
        self.log.append(f"t={now:.1f}: dropping a stroke: {err}")
        self.todo.pop(i)
        return
      if self._clear_of_arm1(env, e, start):
        break
    else:
      if not getattr(self, "yielding", False):
        self.log.append(f"t={now:.1f}: yielding to Arm 1")
        self.yielding = True
      self.hold(0.2, 0.0)
      return
    self.yielding = False
    self.todo.pop(i)
    self.log.append(f"t={now:.1f}: sweeping at x={x:.2f} from y={start[1]:.2f}")
    self._stroke(start)
    self._park()
