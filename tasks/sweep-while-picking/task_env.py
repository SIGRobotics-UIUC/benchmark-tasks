"""The task: your YAM arm sweeps the clutter off the table with a hand broom while a second YAM arm picks red blocks
out of it.

    from task_env import TaskEnv
    env = TaskEnv()
    obs = env.reset()
    while not env.done:
      obs = env.step(action)              # 7 numbers: six joint targets (rad), then the gripper, 0 closed to 1 open
    print(env.score())

The controlled arm is the left one (positive y): on the right of the screenshot, which looks toward the arms.  The
second arm (the right one) is scripted (``sim/partner.py``): it takes the four red blocks to its bin the same way in every
trial.  Control is at 20 Hz, ``RUN_LIMIT`` steps (30 s) at most.  Observations: ``joints`` (13: six joint angles, the gripper
opening 0 to 1, six joint speeds), and two 240 by 320 colour images, ``overhead`` and ``wrist``.
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from cameras import Cameras
from sim import benchmark as b
from sim.benchmark import make_sweep_cfg
from sim.env_cfg import ACTION_SCALE, READY_JOINTS
from sim.ik import ArmIk
from sim.partner import Picker, on_table, pos
from sim.robot import ARM_JOINTS, FINGER_MAX, GRASP_SITE
from sim.scene import LEFT_ARM, RIGHT_ARM

CONTROL_HZ = 20
RUN_LIMIT = 600
"""Control steps: 30 seconds at 20 Hz, about when Arm 1 drops its last block: all the sweeping has to be done while Arm 1
is still working, so waiting for it to finish earns nothing."""
READY = np.array([READY_JOINTS[j] for j in ARM_JOINTS])
PARTNER_SEED = 0
"""Seeds the second arm's start: fixed, so that it moves the same way in every trial."""
HELD = 0.03
"""A red block counts as in Arm 1's hand while its picker holds it and it is within 3 cm of the grasp point."""


def _row(action) -> np.ndarray:
  return np.r_[(np.asarray(action[:6]) - READY) / ACTION_SCALE, np.clip(action[6], 0.0, 1.0)]


class TaskEnv:
  def __init__(self, device: str | None = None) -> None:
    cfg = make_sweep_cfg(LEFT_ARM)
    cfg.episode_length_s = RUN_LIMIT / CONTROL_HZ + 10.0
    cfg.auto_reset = False  # Keep the final state to score.
    del cfg.terminations["arm_collision"]  # Watched here, every simulation step.
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
    self.done = False
    self.contact: str | None = None
    self.partner = Picker(RIGHT_ARM, self.ik, np.random.default_rng(PARTNER_SEED), self.env.step_dt)
    return self._obs()

  def _found(self, sensor: str) -> bool:
    return bool(self.env.scene[sensor].data.found[0, 0] > 0)

  def _touch(self) -> str | None:
    """What touched, if the controlled arm or the broom touched Arm 1 or the block in its hand."""
    if self._found("arm_contact"):
      return "the arms touched"
    if self._found("broom_picker_contact"):
      return "the broom touched Arm 1"
    held = self.partner.holding
    if held is not None:
      robot = self.env.scene[RIGHT_ARM]
      hand = (robot.data.site_pos_w[0, robot.site_names.index(GRASP_SITE)] - self.env.scene.env_origins[0]).cpu().numpy()
      if np.linalg.norm(pos(self.env, held, 0) - hand) < HELD:
        if self._found(f"{held}_sweeper_contact"):
          return "the controlled arm touched the block in Arm 1's hand"
        if self._found(f"{held}_broom_contact"):
          return "the broom touched the block in Arm 1's hand"
    return None

  def cleared(self) -> tuple[int, int]:
    """(items gone, items counted): every clutter item counts, and is gone once it is off the table; a red block counts
    only once it is off the table, so one left on the table, or binned by Arm 1, neither helps nor costs."""
    clutter = sum(not on_table(pos(self.env, n, 0)) for n in b.CLUTTER_NAMES)
    red = sum(not on_table(pos(self.env, n, 0)) for n in b.BLOCK_NAMES)
    return clutter + red, len(b.CLUTTER_NAMES) + red

  finished = None
  """Set by a driver (the demo) to a function saying the controlled arm has finished; otherwise the trial ends as soon
  as everything is gone."""

  def _finished(self) -> bool:
    return self.finished is None or self.finished()

  def _unstable(self) -> str | None:
    for n in (*b.CLUTTER_NAMES, *b.BLOCK_NAMES, "broom"):
      if not np.isfinite(pos(self.env, n, 0)).all():
        return n
    return None

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
      self.contact = self._touch()
      bad = self._unstable()
      if bad:
        raise RuntimeError(f"t={self.ticks * self.env.step_dt:.1f}: the simulation became unstable ({bad}); the trial is invalid")
      gone, counted = self.cleared()
      self.done = self.contact is not None or (gone == counted and self.partner.stage == "done" and self._finished())
    self.done = self.done or self.steps >= RUN_LIMIT
    return self._obs()

  def score(self) -> dict:
    """Items off the table over items counted: the 6 clutter items, plus each red block that is off the table.  0 if the
    controlled arm or the broom ever touched Arm 1 or the block in its hand."""
    gone, counted = self.cleared()
    value = 0.0 if self.contact else gone / counted
    return {"score": round(value, 3), "items_gone": gone, "items_counted": counted,
            "clutter_left": sum(on_table(pos(self.env, n, 0)) for n in b.CLUTTER_NAMES),
            "red_binned": sum(_binned(self.env, n) for n in b.BLOCK_NAMES),
            "red_swept_off": sum(not on_table(pos(self.env, n, 0)) for n in b.BLOCK_NAMES),
            "contact": self.contact, "steps": self.steps}


def _binned(env: ManagerBasedRlEnv, name: str) -> bool:
  from sim.partner import in_bin
  return in_bin(pos(env, name, 0))
