"""The task: your YAM arm keeps a station stocked for a second YAM arm that takes from it on a fixed clock.

    from task_env import TaskEnv
    env = TaskEnv()
    obs = env.reset()
    while not env.done:
      obs = env.step(action)              # 7 numbers: six joint targets (rad), then the gripper, 0 closed to 1 open
    print(env.score())

The controlled arm is the left one (positive y): on the right of the screenshot, which looks toward the arms.  Station B is the small grey tray just on the other arm's side of the middle, with one red block on it; five more lie on the green mat on the controlled arm's side.

The second arm (the right one) works to a fixed routine (``Worker`` in ``sim/partner.py``): From 4 s in, every 11 s, eight times, it takes whatever block is on station B and puts it in the crate on its side; if B is empty it goes away empty-handed.  It does not wait
for the controlled arm or move out of its way, and it moves the same way in every trial.  Control is at 20 Hz,
``RUN_LIMIT`` steps at most.  Observations: ``joints`` (13: six joint angles, the gripper opening 0 to 1, six joint
speeds), and two 240 by 320 colour images, ``overhead`` and ``wrist``.
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from cameras import Cameras
from sim.benchmark import BLOCKS_RESUPPLY, BOX_INNER, CRATE_XY, Z, make_resupply_cfg
from sim.env_cfg import ACTION_SCALE, READY_JOINTS
from sim.ik import ArmIk
from sim.partner import Worker, pos
from sim.robot import ARM_JOINTS, FINGER_MAX
from sim.scene import LEFT_ARM, RIGHT_ARM

CONTROL_HZ = 20
RUN_LIMIT = 2200
"""Control steps: 110 s at 20 Hz, time for the second arm's whole routine."""
READY = np.array([READY_JOINTS[j] for j in ARM_JOINTS])
SETTLED_S = 1.0
"""The trial ends once the score has been 1 this long."""
PARTNER_SEED = 0
"""Seeds the second arm's generator.  Its routine is fixed anyway: it moves the same way in every trial."""


def _row(action) -> np.ndarray:
  return np.r_[(np.asarray(action[:6]) - READY) / ACTION_SCALE, np.clip(action[6], 0.0, 1.0)]


def in_box(env: ManagerBasedRlEnv, name: str, xy, inner, height: float = 0.06) -> bool:
  """Whether ``name``'s middle is within the ``inner`` rectangle round ``xy``, and below ``height`` over the table."""
  p = pos(env, name, 0)
  return bool((np.abs(p[:2] - np.array(xy)) < np.array(inner) / 2).all() and p[2] < Z + height)


def progress(env: ManagerBasedRlEnv, flags: dict) -> tuple[float, dict]:
  crated = [n for n in BLOCKS_RESUPPLY if in_box(env, n, CRATE_XY, BOX_INNER)]
  return len(crated) / len(BLOCKS_RESUPPLY), {"crated": crated}


class TaskEnv:
  def __init__(self, device: str | None = None) -> None:
    cfg = make_resupply_cfg()
    cfg.episode_length_s = RUN_LIMIT / CONTROL_HZ + 10.0
    cfg.auto_reset = False  # Keep the final state to score.
    del cfg.terminations["arm_collision"]  # Watched here, every simulation step, so that the trial ends on the state it reached.
    self.env = ManagerBasedRlEnv(cfg, device=device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    self.dt = 1.0 / CONTROL_HZ
    self.ticks_per_s = round(1.0 / self.env.step_dt)
    self.cameras = Cameras(self.env)
    self.ik = ArmIk()

  def _obs(self) -> dict:
    data = self.env.scene[LEFT_ARM].data
    joints = torch.cat([data.joint_pos[0, :6], (data.joint_pos[0, 6] / FINGER_MAX)[None], data.joint_vel[0, :6]])
    self.cameras.sync()
    return {"joints": joints.cpu().numpy(), "overhead": self.cameras.render("left", "overhead", sync=False),
            "wrist": self.cameras.render("left", "wrist", sync=False)}

  def reset(self) -> dict:
    self.env.reset(seed=0)
    self.steps = self.ticks = self.done_ticks = 0
    self.done = self.touched = False
    self.flags = {"hit": False}
    self.partner = Worker(RIGHT_ARM, self.ik, np.random.default_rng(PARTNER_SEED), self.env.step_dt)
    return self._obs()

  def step(self, action) -> dict:
    if self.done:
      raise RuntimeError("the trial is over; call reset()")
    row = np.asarray(action, dtype=np.float64)
    if row.shape != (7,) or not np.isfinite(row).all():
      raise ValueError("action must be 7 finite numbers")
    own = _row(row)
    self.steps += 1
    # The simulation steps at 50 Hz, the second arm with it; the controlled arm holds its command for the control step.
    until = self.steps * self.ticks_per_s // CONTROL_HZ
    while self.ticks < until and not self.done:
      other = _row(self.partner.act(self.env, 0, self.ticks * self.env.step_dt))
      self.env.step(torch.tensor(np.r_[own, other], dtype=torch.float32, device=self.env.device)[None])
      self.ticks += 1
      self.touched |= bool(self.env.scene["arm_contact"].data.found[0, 0] > 0)
      self.done_ticks = self.done_ticks + 1 if progress(self.env, self.flags)[0] >= 1.0 else 0
      self.done = self.touched or self.done_ticks >= SETTLED_S * self.ticks_per_s
    self.done = self.done or self.steps >= RUN_LIMIT
    return self._obs()

  def score(self) -> dict:
    """The share of the six blocks in the second arm's crate: it crates a block only if station B had one when it came for it.  0 if the arms ever touched."""
    value, details = progress(self.env, self.flags)
    return {"score": 0.0 if self.touched else round(value, 3), **details, "arms_touched": self.touched,
            "steps": self.steps}
