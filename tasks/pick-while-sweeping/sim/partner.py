"""The scripted arms, each reading the scene's true state.

    arm = Sweeper(arm, ik, dt)            # Arm 1: sweeps the table clear (the task's other arm)
    arm = Picker(arm, ik, dt, sweeper)    # the demonstration: puts the red blocks in the bin, out of Arm 1's way
    row = arm.act(env, e, now)            # every step: 6 joint targets (rad) and the gripper, 0 closed to 1 open

The sweeper is sweep-while-picking's demonstration, mirrored, never waiting or keeping clear: it picks up the broom,
makes one stroke toward its own side edge through each band of the table, going straight from one to the next, sweeps
again at any clutter left, and stands the broom back up.  The picker takes the red blocks to its bin one at a time, in a
fixed order, getting out of the sweeper's way whenever it comes close."""

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

  def __init__(self, arm: str, ik: ArmIk, dt: float) -> None:
    self.arm, self.ik, self.dt = arm, ik, dt
    self.side = 1.0 if arm == LEFT_ARM else -1.0
    self.q = READY_Q.copy()
    self.current = READY_Q.copy()
    self.grip = 1.0
    self.queue: list[tuple[np.ndarray, float]] = []
    self.rot = plumb((0.0, 1.0, 0.0))
    self.stage = "wait"
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


class Sweeper(Mover):
  """Arm 1: picks up the broom and sweeps toward its own side edge, one band at a time, going straight from one stroke
  to the next; sweeps again at any clutter left; then stands the broom back up.  It acts as if the other arm weren't
  there."""

  BANDS = ((0.17, 0.08), (0.30, 0.08), (0.42, 0.08))
  """(grip x, start y) of the strokes, one per band, over this arm's half of the table; y is toward this arm.  None
  nearer this arm's base than x = 0.17: the broom caught on it."""
  CLEARANCE = 0.010
  """The bristles' tips ride this far above the table: every item is at least 3 cm tall, and bristles dragging on the
  table caught at its edge and twisted the broom out of the hand."""
  EDGE = TABLE_SIZE[1] / 2 + 0.02
  """Strokes end with the broom's middle this far out: its front face 4.5 cm past the edge."""
  PACE = 3.0
  """Seconds per metre of stroke."""
  LIFT = 0.08
  """How high the bristles come straight down onto the table from, at the start of a stroke, and go up at its end: with
  less, the move straight on to the next stroke dipped and knocked blocks about."""

  def __init__(self, arm: str, ik: ArmIk, dt: float) -> None:
    super().__init__(arm, ik, dt)
    self.rot = leaned((0.0, 1.0, 0.0))  # The jaw across the grip, the broom's length along x.
    self.todo = list(self.BANDS)
    self.extra = 0

  def _start(self, x: float, y0: float) -> np.ndarray:
    low = b.Z + props.GRIP_HEIGHT + self.CLEARANCE
    for y in np.arange(y0, 0.36, 0.02):
      start = np.array([x, self.side * y, low])
      try:
        self.solve(start + [0, 0, self.LIFT])
        self.solve(start)
      except RuntimeError:
        continue
      return start
    raise RuntimeError(f"no reachable start at x={x:.2f}")

  def _stroke(self, start: np.ndarray) -> None:
    end = np.array([start[0], self.side * self.EDGE, start[2]])
    self.glide(self.solve(start + [0, 0, self.LIFT]), 1.2, 0.0)
    self.line(start + [0, 0, self.LIFT], start, 0.5, 0.0)
    self.line(start, end, self.PACE * abs(end[1] - start[1]), 0.0)
    self.line(end, end + [0, 0, self.LIFT], 0.4, 0.0)

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
      return
    if not self.todo:
      left = self._left(env, e)
      if not left or self.extra >= 12:
        self._put_back()
        self.stage = "done"
        return
      p = min(left, key=lambda p: self.side * p[1])  # The item farthest from the edge: sweeping it takes others with it.
      self.todo.append((float(np.clip(p[0], 0.17, 0.42)), max(self.side * p[1] - 0.07, -0.12)))
      self.extra += 1
    x, y0 = self.todo.pop(0)
    try:
      start = self._start(x, y0)
    except RuntimeError as err:
      self.log.append(f"t={now:.1f}: dropping a stroke: {err}")
      return
    self.log.append(f"t={now:.1f}: sweeping at x={x:.2f} from y={start[1]:.2f}")
    self._stroke(start)  # Then straight on to the next: it never waits for, or keeps clear of, the other arm.


def arm_points(ik: ArmIk, q: np.ndarray, arm: str, broom: bool = False) -> np.ndarray:
  """Points along an arm at joints ``q``: each link's origin, three points between each pair, and the grasp point; with
  ``broom``, also points over the broom it holds (20 cm along x, its back 5 to 8 cm below the grip)."""
  ik._set(q)
  data, model = ik.data, ik.model
  chain = [data.xpos[model.body(n).id] for n in ("link2", "link3", "link4", "link5", "link6", "gripper")]
  grasp = data.site_xpos[ik.site].copy()
  chain.append(grasp)
  chain += [u + (v - u) * f for u, v in zip(chain[:-1], chain[1:]) for f in (0.25, 0.5, 0.75)]
  if broom:
    chain += [grasp + [dx, 0.0, dz] for dx in np.linspace(-0.10, 0.10, 5) for dz in (-0.05, -0.08)]
  return np.array(chain) + np.array(BASE_POSE[arm][0])


class Picker(Mover):
  """The demonstration: the red blocks, one by one, into the bin: the two deep in Arm 1's side first, then the two near
  the middle.  It doesn't wait for Arm 1, but
  watches it: when Arm 1 or its broom is about to come within ``KEEP_OUT``, it gets out of the way over its bin, dropping
  any block it holds (that block is tried again last).  If the broom moves the block it is reaching for before the
  gripper closes, it backs off and goes for the next."""

  CARRY = 0.15
  """Height above the table at which a block is carried and dropped over the bin."""
  MOVED = 0.02
  """A block that has moved this far since the reach began has been taken by the broom."""
  KEEP_OUT = 0.12
  """Between points on the two arms' centre lines: the link shells and the gripper's width take up most of it."""
  SOON = 0.6
  """Seconds ahead it looks, along both arms' planned motion, for Arm 1 coming."""

  def __init__(self, arm: str, ik: ArmIk, dt: float, sweeper: Sweeper) -> None:
    super().__init__(arm, ik, dt)
    self.sweeper = sweeper
    self.reaching = 0
    self.todo = list(b.BLOCK_NAMES)
    self.holding: str | None = None
    self.stage = "next"
    self.ordered = False

  def _pick(self, env: ManagerBasedRlEnv, e: int, name: str) -> list[tuple[np.ndarray, float]] | None:
    """The rows taking ``name`` into the bin, or None if the arm can't reach it."""
    p = pos(env, name, e)
    # The jaw across the block's faces, whichever pair is nearest across the table (y).
    yaw = block_yaw(env, name, e)
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    q0, queue0 = self.q.copy(), self.queue
    self.queue = []
    try:
      self.rot = leaned((-np.sin(yaw), np.cos(yaw), 0.0))
      grasp = p + [0.0, 0.0, 0.010]  # Any lower, and at full reach the leaning fingers pressed on the table.
      for rise in (0.10, 0.08, 0.06):  # As high above the block as the arm reaches there.
        above = grasp + [0, 0, rise]
        try:
          q_above = self.solve(above)
          self.solve(grasp)
        except RuntimeError:
          continue
        break
      else:
        return None
      self.glide(q_above, 1.2, 1.0)
      self.line(above, grasp, 0.8, 1.0)  # Any faster, and the fingers caught the cube's edge.
      self.reach = len(self.queue)  # Rows until the gripper starts to close.
      self.hold(0.7, 0.0)  # The gripper takes about that long to close.
      self.line(grasp, above, 0.8, 0.0)
      self.rot = leaned((0.0, 1.0, 0.0))
      # Each block to its own spot in the bin, so they don't stack; the bin's middle if the arm can't reach the spot.
      spot = [(0.04, 0.035), (0.04, -0.035), (0.0, 0.035), (0.0, -0.035)][b.BLOCK_NAMES.index(name) % 4]
      try:
        q = self.solve([b.BIN_XY[0] + spot[0], b.BIN_XY[1] + spot[1], b.Z + self.CARRY])
      except RuntimeError:
        q = self.solve([b.BIN_XY[0], b.BIN_XY[1], b.Z + self.CARRY])
      self.glide(q, 1.4, 0.0)  # Any faster, and the block slipped out.
      self.hold(0.8, 1.0)  # Time to open and for the block to land.
      return self.queue
    except RuntimeError:
      return None
    finally:
      self.queue, self.q = queue0, q0

  def plan(self, env: ManagerBasedRlEnv, e: int, now: float) -> None:
    if self.stage == "retreat":
      self.stage = "next"
      return
    if self.stage == "carry":
      if not in_bin(pos(env, self.holding, e)):
        self.log.append(f"t={now:.1f}: lost {self.holding}")
      self.holding = None
      self.stage = "next"
      return
    if self.stage != "next" or now < 0.1:  # Until the simulation has stepped, poses are not yet read back.
      return
    if not self.ordered:
      # The two deep in Arm 1's side first, nearest the bases first, then the two near the middle.
      self.todo.sort(key=lambda n: (round(self.side * pos(env, n, e)[1], 2), pos(env, n, e)[0]))
      self.ordered = True
    self.todo = [n for n in self.todo if on_table(pos(env, n, e)) and not in_bin(pos(env, n, e))]
    if not self.todo:
      # Done: wait high over the bin, out of the way.
      self.rot = leaned((0.0, 1.0, 0.0))
      self.glide(self.solve([b.BIN_XY[0], b.BIN_XY[1], b.Z + 0.20]), 1.5, 1.0)
      self.stage = "done"
      return
    # The next block, in order.
    name = self.todo.pop(0)
    rows = self._pick(env, e, name)
    if rows is None:
      self.log.append(f"t={now:.1f}: cannot reach {name}; skipping it")
      return
    self.log.append(f"t={now:.1f}: taking {name}")
    self.queue += rows
    self.q = rows[-1][0]
    self.holding, self.target, self.reaching = name, pos(env, name, e), self.reach
    self.stage = "carry"

  def _coming(self, env: ManagerBasedRlEnv, e: int) -> bool:
    """Whether Arm 1 or its broom comes within ``KEEP_OUT`` of this arm over the next ``SOON`` seconds."""
    other = self.sweeper
    theirs = [env.scene[other.arm].data.joint_pos[e, :6].cpu().numpy()]
    theirs += [q for q, _ in other.queue[:int(self.SOON / other.dt):5]]
    ours = [env.scene[self.arm].data.joint_pos[e, :6].cpu().numpy()]
    ours += [q for q, _ in self.queue[:int(self.SOON / self.dt):3]]
    a = np.concatenate([arm_points(self.ik, q, other.arm, broom=True) for q in theirs])
    m = np.concatenate([arm_points(self.ik, q, self.arm) for q in ours])
    return np.linalg.norm(m[:, None] - a[None], axis=-1).min() < self.KEEP_OUT

  def act(self, env: ManagerBasedRlEnv, e: int, now: float) -> np.ndarray:
    if self.stage in ("next", "carry") and self._coming(env, e):
      # Out of the way: up and over the bin, letting go of any block.
      self.log.append(f"t={now:.1f}: Arm 1 coming; getting out of the way")
      if self.holding is not None:
        self.todo.append(self.holding)
      self.queue.clear()
      self.q, self.reaching, self.holding, self.stage = self.current.copy(), 0, None, "retreat"
      self.rot = leaned((0.0, 1.0, 0.0))
      self.glide(self.solve([b.BIN_XY[0], b.BIN_XY[1], b.Z + 0.20]), 1.2, 1.0)
    if self.reaching:
      self.reaching -= 1
      if np.linalg.norm(pos(env, self.holding, e) - self.target) > self.MOVED:
        # The broom got there first: back off over the bin and go for the next block.
        self.log.append(f"t={now:.1f}: the broom took {self.holding}; moving on")
        self.queue.clear()
        self.q, self.reaching, self.holding, self.stage = self.current.copy(), 0, None, "next"
        self.rot = leaned((0.0, 1.0, 0.0))
        self.glide(self.solve([b.BIN_XY[0], b.BIN_XY[1], b.Z + self.CARRY]), 1.4, 1.0)
    return super().act(env, e, now)
