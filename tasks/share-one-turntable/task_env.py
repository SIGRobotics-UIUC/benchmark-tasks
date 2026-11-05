"""The task: your YAM arm gets its cube off a turntable it shares with a second YAM arm, keeping to its own side.

    from task_env import TaskEnv
    env = TaskEnv()
    obs = env.reset()
    while not env.done:
      obs = env.step(action)              # 7 numbers: six joint targets (rad), then the gripper, 0 closed to 1 open
    print(env.score())

The controlled arm is the left one (positive y): on the right of the screenshot, which looks toward the arms.  Its cube
is the green one, which starts on the turntable on the other arm's side; its tray is the green one on its own side.  Its
hand must stay on its own half of the table: no more than ``CENTRE_LINE_SLACK`` past the centre line.  The second arm
(the right one) is scripted (``sim/partner.py``): it reads the scene's true state; while its own cube is not on its side
and the controlled arm's hand is away from the turntable, it turns the turntable a quarter turn anticlockwise by a peg on
its own half, once, or again if the controlled arm's hand has kept away from the turntable for 8 s; then it puts its
cube in its tray, the same way in every trial.  Control is at 20 Hz, ``RUN_LIMIT`` steps at most.  Observations:
``joints`` (13: six joint angles, the gripper opening 0 to 1, six joint speeds), and two 240 by 320 colour images,
``overhead`` and ``wrist``.
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from cameras import Cameras
from sim.benchmark import CENTRE_LINE_SLACK, SPIN_ITEMS, SPIN_TRAY_INNER, SPIN_TRAY_XY, Z, make_turntable_cfg
from sim.env_cfg import ACTION_SCALE, READY_JOINTS
from sim.ik import ArmIk
from sim.partner import Turner, pos, site
from sim.robot import ARM_JOINTS, FINGER_MAX, GRASP_SITE
from sim.scene import LEFT_ARM, RIGHT_ARM

CONTROL_HZ = 20
RUN_LIMIT = 3600
"""Control steps: 3 minutes at 20 Hz."""
READY = np.array([READY_JOINTS[j] for j in ARM_JOINTS])
SETTLED_S = 1.0
"""The trial ends once the controlled arm's cube has been in its tray this long."""
PARTNER_SEED = 0
"""Seeds the second arm's random waits: fixed, so that it moves the same way in every trial."""


def _row(action) -> np.ndarray:
  return np.r_[(np.asarray(action[:6]) - READY) / ACTION_SCALE, np.clip(action[6], 0.0, 1.0)]


def in_tray(env: ManagerBasedRlEnv, cube: str) -> bool:
  owner = SPIN_ITEMS[cube][0]
  p = pos(env, cube, 0)
  return bool((np.abs(p[:2] - np.array(SPIN_TRAY_XY[owner])) < np.array(SPIN_TRAY_INNER) / 2).all() and p[2] < Z + 0.06)


class TaskEnv:
  def __init__(self, device: str | None = None) -> None:
    cfg = make_turntable_cfg()
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
    self.steps = self.ticks = self.home_ticks = 0
    self.done = self.touched = self.crossed = False
    self.partner = Turner(RIGHT_ARM, self.ik, np.random.default_rng(PARTNER_SEED), self.env.step_dt)
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
      self.crossed |= bool(site(self.env, LEFT_ARM, GRASP_SITE, 0)[1] < -CENTRE_LINE_SLACK)
      self.home_ticks = self.home_ticks + 1 if in_tray(self.env, "left_cube") else 0
      self.done = self.touched or self.crossed or self.home_ticks >= SETTLED_S * self.ticks_per_s
    self.done = self.done or self.steps >= RUN_LIMIT
    return self._obs()

  def score(self) -> dict:
    """1 if the controlled arm's cube is in its tray at the end, else 0; 0 if its hand went past the centre line, or if
    the arms ever touched."""
    home = in_tray(self.env, "left_cube")
    value = 0.0 if self.touched or self.crossed else float(home)
    return {"score": value, "cube_in_tray": home, "hand_crossed_centre": self.crossed, "arms_touched": self.touched,
            "other_cube_in_tray": in_tray(self.env, "right_cube"), "steps": self.steps}
