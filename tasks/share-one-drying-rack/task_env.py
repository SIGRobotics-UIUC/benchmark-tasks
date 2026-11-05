"""The task: your YAM arm puts its cups into a drying rack it shares with a second YAM arm.

    from task_env import TaskEnv
    env = TaskEnv()
    obs = env.reset()
    while not env.done:
      obs = env.step(action)              # 7 numbers: six joint targets (rad), then the gripper, 0 closed to 1 open
    print(env.score())

The controlled arm is the left one (positive y): on the right of the screenshot, which looks toward the arms.  Its two
green cups stand on the green mat on its side.  The rack has four bays, one cup to a bay, for the four cups on the
table.  The second arm (the right one) is scripted (``sim/partner.py``): it reads the scene's true state and puts its two
blue cups into free bays, keeping its hand away from the rack while the controlled arm's hand is there, the same way in
every trial.  Control is at 20 Hz, ``RUN_LIMIT`` steps at most.  Observations: ``joints`` (13: six joint angles, the
gripper opening 0 to 1, six joint speeds), and two 240 by 320 colour images, ``overhead`` and ``wrist``.
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from cameras import Cameras
from sim.benchmark import CUPS, make_drying_rack_cfg
from sim.env_cfg import ACTION_SCALE, READY_JOINTS
from sim.ik import ArmIk
from sim.partner import Racker, in_rack
from sim.robot import ARM_JOINTS, FINGER_MAX
from sim.scene import LEFT_ARM, RIGHT_ARM

CONTROL_HZ = 20
RUN_LIMIT = 3600
"""Control steps: 3 minutes at 20 Hz."""
READY = np.array([READY_JOINTS[j] for j in ARM_JOINTS])
SETTLED_S = 1.0
"""The trial ends once both of the controlled arm's cups have been in the rack this long."""
PARTNER_SEED = 0
"""Seeds the second arm's random waits: fixed, so that it moves the same way in every trial."""


def _row(action) -> np.ndarray:
  return np.r_[(np.asarray(action[:6]) - READY) / ACTION_SCALE, np.clip(action[6], 0.0, 1.0)]


def racked(env: ManagerBasedRlEnv, arm: str) -> list[str]:
  """``arm``'s cups that are in the rack."""
  inside = in_rack(env, 0)
  return sorted(n for n, (owner, _, _) in CUPS.items() if owner == arm and inside[n])


def mine(arm: str) -> int:
  return sum(owner == arm for owner, _, _ in CUPS.values())


class TaskEnv:
  def __init__(self, device: str | None = None) -> None:
    cfg = make_drying_rack_cfg()
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
    self.steps = self.ticks = self.racked_ticks = 0
    self.done = self.touched = False
    self.partner = Racker(RIGHT_ARM, self.ik, np.random.default_rng(PARTNER_SEED), self.env.step_dt)
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
      self.racked_ticks = self.racked_ticks + 1 if len(racked(self.env, LEFT_ARM)) == mine(LEFT_ARM) else 0
      self.done = self.touched or self.racked_ticks >= SETTLED_S * self.ticks_per_s
    self.done = self.done or self.steps >= RUN_LIMIT
    return self._obs()

  def score(self) -> dict:
    """The share of the controlled arm's cups down in a bay of the rack at the end; 0 if the arms ever touched."""
    inside = racked(self.env, LEFT_ARM)
    value = 0.0 if self.touched else len(inside) / mine(LEFT_ARM)
    return {"score": round(value, 3), "in_rack": inside, "arms_touched": self.touched,
            "other_cups_in_rack": len(racked(self.env, RIGHT_ARM)), "steps": self.steps}
