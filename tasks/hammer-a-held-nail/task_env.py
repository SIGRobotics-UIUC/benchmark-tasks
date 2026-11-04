"""The task: your YAM arm hammers a nail into a post while a second YAM arm holds it.

    from task_env import TaskEnv
    env = TaskEnv()
    obs = env.reset()
    while not env.done:
      obs = env.step(action)              # 7 numbers: six joint targets (rad), then the gripper, 0 closed to 1 open
    print(env.score())

The controlled arm is the left one (positive y): on the right of the screenshot, which looks toward the arms.  The
second arm (the right one) is scripted (``sim/partner.py``): it reads the scene's true state and does its part of the job
the same way in every trial.  Control is at 20 Hz, ``RUN_LIMIT`` steps at most.  Observations: ``joints`` (13: six joint
angles, the gripper opening 0 to 1, six joint speeds), and two 240 by 320 colour images, ``overhead`` and ``wrist``.
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse

from cameras import Cameras
from sim import props
from sim.benchmark import make_held_nail_cfg
from sim.env_cfg import ACTION_SCALE, READY_JOINTS
from sim.ik import ArmIk
from sim.partner import Holder, depth
from sim.robot import ARM_JOINTS, FINGER_MAX, GRASP_SITE
from sim.scene import LEFT_ARM, RIGHT_ARM

CONTROL_HZ = 20
RUN_LIMIT = 3600
"""Control steps: 3 minutes at 20 Hz."""
READY = np.array([READY_JOINTS[j] for j in ARM_JOINTS])
DRIVE = props.SPIKE_DRIVE
"""The nail is in when it has gone this far into the post: 4 cm."""

PARTNER_SEED = 0
"""Seeds the second arm's random start: fixed, so that it moves the same way in every trial."""


def _row(action) -> np.ndarray:
  return np.r_[(np.asarray(action[:6]) - READY) / ACTION_SCALE, np.clip(action[6], 0.0, 1.0)]


class TaskEnv:
  def __init__(self, device: str | None = None) -> None:
    cfg = make_held_nail_cfg(RIGHT_ARM)
    cfg.episode_length_s = RUN_LIMIT / CONTROL_HZ + 10.0
    cfg.auto_reset = False  # Keep the final state to score.
    for term in ("arm_collision", "hammer_hit_holder"):
      del cfg.terminations[term]  # Watched here, every simulation step, so that the trial ends on the state it reached.
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
    self.steps = self.ticks = 0
    self.done = self.touched = self.hit = False
    self.credited = self.deepest = 0.0
    self.left_at = None  # How deep the nail was when the second arm let go of it, if it was the holder.
    self.partner = Holder(RIGHT_ARM, self.ik, np.random.default_rng(PARTNER_SEED), self.env.step_dt)
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
      vector = torch.tensor(np.r_[own, other], dtype=torch.float32, device=self.env.device)[None]
      self.env.step(vector)
      self.ticks += 1
      self.touched |= bool(self.env.scene["arm_contact"].data.found[0, 0] > 0)
      self.hit |= bool(self.env.scene["hammer_holder_contact"].data.found[0, 0] > 0)
      driven = depth(self.env, 0)
      if self.left_at is None and self.partner.stage in ("release", "done") and hasattr(self.partner, "upright"):
        self.left_at = driven
      if self.left_at is not None and driven > self.deepest and self._holding_hammer():
        self.credited += driven - max(self.deepest, self.left_at)
      self.deepest = max(self.deepest, driven)
      self.done = self.touched or self.hit or driven >= DRIVE
    self.done = self.done or self.steps >= RUN_LIMIT
    return self._obs()

  def _holding_hammer(self) -> bool:
    """The controlled arm has the hammer: its grasp site within 5 cm of the handle and its fingers closing."""
    hammer = self.env.scene["hammer"].data
    robot = self.env.scene[LEFT_ARM]
    here = robot.data.site_pos_w[:1, robot.site_names.index(GRASP_SITE)]
    local = quat_apply_inverse(hammer.root_link_quat_w[:1], here - hammer.root_link_pos_w[:1])[0]
    off_axis = float(local[1:].norm() + (local[0].abs() - props.HAMMER_HANDLE / 2).clamp(min=0.0))
    return off_axis < 0.05 and float(robot.data.joint_pos[0, robot.joint_names.index("joint7")]) < 0.8 * FINGER_MAX

  def score(self) -> dict:
    """What the nail went in after the second arm let go of it, while the controlled arm held the hammer, over what was
    left to go to 4 cm, up to 1.  (The second arm presses the nail in itself as it holds it; that does not count.)  0 if
    the arms ever touched, or the hammer ever touched the holding arm, or the second arm never let go."""
    driven = depth(self.env, 0)
    value = 0.0
    if self.left_at is not None and not (self.touched or self.hit):
      value = min(self.credited / max(DRIVE - self.left_at, 1e-6), 1.0)
    return {"score": round(value, 3), "nail_depth_mm": round(1000 * driven, 1),
            "let_go_at_mm": None if self.left_at is None else round(1000 * self.left_at, 1),
            "hammered_mm": round(1000 * self.credited, 1), "arms_touched": self.touched,
            "hammer_hit_holding_arm": self.hit, "steps": self.steps}
