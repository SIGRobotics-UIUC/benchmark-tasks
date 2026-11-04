"""The task: your YAM arm drives its own nail with the one hammer it shares with a second YAM arm.

    from task_env import TaskEnv
    env = TaskEnv()
    obs = env.reset()
    while not env.done:
      obs = env.step(action)              # 7 numbers: six joint targets (rad), then the gripper, 0 closed to 1 open
    print(env.score())

The controlled arm is the left one (positive y): on the right of the screenshot, which looks toward the arms.  The
second arm (the right one) is scripted (``sim/partner.py``): it reads the scene's true state, waits for the hammer to
be free, drives its own nail and puts the hammer back, the same way in every trial.  Control is at 20 Hz, ``RUN_LIMIT``
steps at most.  Observations: ``joints`` (13: six joint angles, the gripper opening 0 to 1, six joint speeds), and two
240 by 320 colour images, ``overhead`` and ``wrist``.
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from cameras import Cameras
from sim import props
from sim.benchmark import Z, make_shared_hammer_cfg
from sim.env_cfg import ACTION_SCALE, READY_JOINTS
from sim.ik import ArmIk
from sim.partner import Sharer, holding, nail_depth, pos
from sim.robot import ARM_JOINTS, FINGER_MAX
from sim.scene import LEFT_ARM, RIGHT_ARM

CONTROL_HZ = 20
RUN_LIMIT = 3600
"""Control steps: 3 minutes at 20 Hz."""
READY = np.array([READY_JOINTS[j] for j in ARM_JOINTS])
HOME = props.NAIL_STANDOUT - 0.008
"""The nail is home when its head is within 8 mm of the wood: driven 17 mm."""
MOVED = 0.0005
"""A step in which the other arm's nail goes in by less than this does not count against the controlled arm."""
PARTNER_SEED = 0
"""Seeds the second arm's random waits: fixed, so that it moves the same way in every trial."""


def _row(action) -> np.ndarray:
  return np.r_[(np.asarray(action[:6]) - READY) / ACTION_SCALE, np.clip(action[6], 0.0, 1.0)]


class TaskEnv:
  def __init__(self, device: str | None = None) -> None:
    cfg = make_shared_hammer_cfg()
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
    self.steps = self.ticks = 0
    self.done = self.touched = self.wrong_nail = False
    self.credited = 0.0
    self.deepest = {arm: nail_depth(self.env, arm, 0) for arm in (LEFT_ARM, RIGHT_ARM)}
    self.partner = Sharer(RIGHT_ARM, self.ik, np.random.default_rng(PARTNER_SEED), self.env.step_dt)
    return self._obs()

  def _account(self) -> None:
    """Credit new depth on the controlled arm's nail while it holds the hammer; mark it for the other nail if that one
    goes in while the controlled arm holds the hammer and the other arm does not."""
    mine, theirs = holding(self.env, LEFT_ARM, 0), holding(self.env, RIGHT_ARM, 0)
    for arm in (LEFT_ARM, RIGHT_ARM):
      depth = nail_depth(self.env, arm, 0)
      moved = max(depth - self.deepest[arm], 0.0)
      self.deepest[arm] = max(self.deepest[arm], depth)
      if arm == LEFT_ARM and mine:
        self.credited += moved
      if arm == RIGHT_ARM and moved > MOVED and mine and not theirs:
        self.wrong_nail = True

  def _hammer_down(self) -> bool:
    return not holding(self.env, LEFT_ARM, 0) and pos(self.env, "hammer", 0)[2] < Z + 0.05

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
      self._account()
      self.done = self.touched or self.wrong_nail or (self.credited >= HOME and self._hammer_down())
    self.done = self.done or self.steps >= RUN_LIMIT
    return self._obs()

  def score(self) -> dict:
    """The new depth the controlled arm's nail went in while that arm held the hammer, over 17 mm; halved if the arm
    still holds the hammer at the end (the other arm needs it); 0 if the arms ever touched or the controlled arm drove
    the other arm's nail."""
    progress = min(self.credited / HOME, 1.0)
    down = bool(self._hammer_down())
    value = 0.0 if self.touched or self.wrong_nail else progress * (1.0 if down else 0.5)
    return {"score": round(value, 3), "own_nail_mm": round(1000 * self.credited, 1), "hammer_put_down": down,
            "arms_touched": self.touched, "drove_other_nail": self.wrong_nail, "steps": self.steps,
            "other_nail_mm": round(1000 * nail_depth(self.env, RIGHT_ARM, 0), 1)}
