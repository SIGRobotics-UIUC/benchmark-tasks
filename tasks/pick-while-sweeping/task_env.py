"""The task: your YAM arm puts four red blocks in its bin while a second YAM arm sweeps the table clear with a hand
broom.

    from task_env import TaskEnv
    env = TaskEnv()
    obs = env.reset()
    while not env.done:
      obs = env.step(action)              # 7 numbers: six joint targets (rad), then the gripper, 0 closed to 1 open
    print(env.score())

The controlled arm is the left one (positive y): on the right of the screenshot, which looks toward the arms.  The
second arm (the right one) is scripted (``sim/partner.py``): it sweeps the table on a fixed, open-loop routine, the same
way in every trial.  Control is at 20 Hz, ``RUN_LIMIT`` steps (30 s) at most.  Observations: ``joints`` (13: six joint
angles, the gripper opening 0 to 1, six joint speeds), and two 240 by 320 colour images, ``overhead`` and ``wrist``.
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from cameras import Cameras
from sim import benchmark as b
from sim.benchmark import make_pick_cfg
from sim.env_cfg import ACTION_SCALE, READY_JOINTS
from sim.ik import ArmIk
from sim.partner import Sweeper, in_bin, on_table, pos, swept
from sim.robot import ARM_JOINTS, FINGER_MAX, GRASP_SITE
from sim.scene import LEFT_ARM, RIGHT_ARM

CONTROL_HZ = 20
RUN_LIMIT = 600
"""Control steps: 30 seconds at 20 Hz."""
READY = np.array([READY_JOINTS[j] for j in ARM_JOINTS])
HELD = 0.03
"""A red block counts as in the controlled arm's hand while it is within 3 cm of the grasp point."""


def _row(action) -> np.ndarray:
  return np.r_[(np.asarray(action[:6]) - READY) / ACTION_SCALE, np.clip(action[6], 0.0, 1.0)]


class TaskEnv:
  def __init__(self, device: str | None = None) -> None:
    cfg = make_pick_cfg(LEFT_ARM)
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
    self.partner = Sweeper(RIGHT_ARM, ArmIk(), self.env.step_dt)  # Its own IK, so its plan never depends on yours.
    return self._obs()

  def _found(self, sensor: str) -> bool:
    return bool(self.env.scene[sensor].data.found[0, 0] > 0)

  def _touch(self) -> str | None:
    """What touched, if the controlled arm or a block in its hand touched Arm 1 or its broom."""
    if self._found("arm_contact"):
      return "the arms touched"
    if self._found("broom_picker_contact"):
      return "the controlled arm touched the broom"
    hand = self._hand()
    for name in b.BLOCK_NAMES:
      if np.linalg.norm(pos(self.env, name, 0) - hand) < HELD:
        if self._found(f"{name}_sweeper_contact"):
          return "the block in the controlled arm's hand touched Arm 1"
        if self._found(f"{name}_broom_contact"):
          return "the block in the controlled arm's hand touched the broom"
    return None

  def _hand(self) -> np.ndarray:
    robot = self.env.scene[LEFT_ARM]
    return (robot.data.site_pos_w[0, robot.site_names.index(GRASP_SITE)] - self.env.scene.env_origins[0]).cpu().numpy()

  def _resting(self, name: str) -> bool:
    """In the bin and let go: a block still within 3 cm of the grasp point is in the hand, not resting in the bin."""
    p = pos(self.env, name, 0)
    return bool(in_bin(p) and np.linalg.norm(p - self._hand()) >= HELD)

  def binned(self) -> int:
    return sum(self._resting(n) for n in b.BLOCK_NAMES)

  def _settled(self) -> bool:
    """Every red block is resting in the bin, swept past the line or off the table: nothing left the arm can bin."""
    return all(self._resting(n) or swept(pos(self.env, n, 0)) or not on_table(pos(self.env, n, 0))
               for n in b.BLOCK_NAMES)

  finished = None
  """Set by a driver (the demo) to a function saying the controlled arm has finished."""

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
      self.done = self.contact is not None or (self._settled() and self.partner.stage == "done" and not self.partner.queue and self._finished())
    self.done = self.done or self.steps >= RUN_LIMIT
    return self._obs()

  def score(self) -> dict:
    """The share of the four red blocks resting in the bin, let go of.  0 if the controlled arm, or a block in its hand, ever touched Arm 1
    or its broom."""
    binned = self.binned()
    value = 0.0 if self.contact else binned / len(b.BLOCK_NAMES)
    return {"score": round(value, 3), "red_binned": binned,
            "red_swept": sum(swept(pos(self.env, n, 0)) for n in b.BLOCK_NAMES),
            "clutter_left": sum(not swept(pos(self.env, n, 0)) for n in b.CLUTTER_NAMES),
            "contact": self.contact, "steps": self.steps}
