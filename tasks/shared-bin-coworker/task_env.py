"""The task: your YAM arm takes the fruit out of a shared bin while a second YAM arm takes the utensils out of it.

    from task_env import TaskEnv
    env = TaskEnv()                       # one fixed layout, the second arm replaying its recorded run
    obs = env.reset()
    while not env.done:
      obs = env.step(action)              # 7 numbers: six joint targets (rad), then the gripper, 0 closed to 1 open
    print(env.score())

The controlled arm is the one on the left of the table seen from the arms' side (positive y); the second arm is on the
right.  Control is at 20 Hz; ``RUN_LIMIT`` steps at most.  Observations: ``joints`` (13: six joint angles, the gripper
opening 0 to 1, six joint speeds), and two 240 by 320 colour images, ``overhead`` and ``wrist``.

The second arm does its own job and does not react to yours: it replays, joint by joint, a recorded run in which it
cleared its utensils, so every trial sees it make the same moves at the same times.
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from cameras import Cameras
from sim.benchmark import FRUIT, TRAY_INNER, TRAY_XY, UTENSILS, make_shared_bin_cfg
from sim.env_cfg import ACTION_SCALE, READY_JOINTS
from sim.robot import ARM_JOINTS, FINGER_MAX
from sim.scene import LEFT_ARM, RIGHT_ARM, TABLE_TOP
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
LAYOUT_SEED = 0
"""The one starting layout: which slot of the bin each object starts in, as in the screenshot."""
CONTROL_HZ = 20
RUN_LIMIT = 600
"""Control steps: 30 s at 20 Hz.  The other arm's recorded run lasts about 26 s, so waiting for it to finish before
starting leaves too little time to do the task: the arm under test has to work while the other one does."""
READY = np.array([READY_JOINTS[j] for j in ARM_JOINTS])
OFF_TABLE_Z = TABLE_TOP - 0.05


REPLAY_HZ = 50
"""The other arm's recording has a target every 20 ms.  The simulation steps at that rate inside each 50 ms control
step, so the other arm gets every recorded target at its recorded time, while the arm under test holds its command for
the whole control step.  Sampling the recording at 20 Hz instead changed the other arm's grasps enough to miss."""


class TaskEnv:
  def __init__(self, device: str | None = None) -> None:
    cfg = make_shared_bin_cfg(num_envs=1)
    cfg.decimation = round(1.0 / (REPLAY_HZ * cfg.sim.mujoco.timestep))
    cfg.episode_length_s = RUN_LIMIT / CONTROL_HZ
    cfg.auto_reset = False  # Keep the final state to score: an automatic reset would put everything back in the bin.
    self.env = ManagerBasedRlEnv(cfg, device=device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    self.dt = 1.0 / CONTROL_HZ
    replay = np.load(HERE / "data" / "recorded_rows.npz")
    self.partner_rows = replay["right"]
    self.cameras = Cameras(self.env)

  def _partner_row(self) -> np.ndarray:
    return self.partner_rows[min(self.ticks, len(self.partner_rows) - 1)]

  def _obs(self) -> dict:
    data = self.env.scene[LEFT_ARM].data
    joints = torch.cat([data.joint_pos[0, :6], (data.joint_pos[0, 6] / FINGER_MAX)[None], data.joint_vel[0, :6]])
    self.cameras.sync()
    return {"joints": joints.cpu().numpy(), "overhead": self.cameras.render("left", "overhead", sync=False),
            "wrist": self.cameras.render("left", "wrist", sync=False)}

  def reset(self) -> dict:
    self.env.reset(seed=LAYOUT_SEED)
    self.steps, self.ticks, self.done, self.touched = 0, 0, False, False
    self.utensil_lost = False
    return self._obs()

  def step(self, action) -> dict:
    if self.done:
      raise RuntimeError("the trial is over; call reset()")
    row = np.asarray(action, dtype=np.float64)
    if row.shape != (7,) or not np.isfinite(row).all():
      raise ValueError("action must be 7 finite numbers")
    own = np.r_[(row[:6] - READY) / ACTION_SCALE, np.clip(row[6], 0, 1)]
    self.steps += 1
    # The simulation ticks at REPLAY_HZ: 2 or 3 ticks per control step, alternating, 2.5 on average.
    until = self.steps * REPLAY_HZ // CONTROL_HZ
    while self.ticks < until and not self.done:
      partner = self._partner_row()
      vector = np.r_[own, (partner[:6] - READY) / ACTION_SCALE, partner[6]]
      _, _, terminated, truncated, _ = self.env.step(torch.tensor(vector, dtype=torch.float32, device=self.env.device)[None])
      self.ticks += 1
      self.touched = bool(self.env.termination_manager.get_term("arm_collision")[0])
      if not self.touched:
        self._measure()
      self.done = bool(terminated[0] or truncated[0])
    self.done = self.done or self.steps >= RUN_LIMIT
    return self._obs()

  def _pos(self, name: str) -> np.ndarray:
    return (self.env.scene[name].data.root_link_pos_w[0] - self.env.scene.env_origins[0]).cpu().numpy()

  def _measure(self) -> None:
    centre, half = np.array(TRAY_XY[LEFT_ARM]), np.array(TRAY_INNER) / 2
    self.in_tray = [bool((np.abs(self._pos(f)[:2] - centre) < half).all() and self._pos(f)[2] < TABLE_TOP + 0.08) for f in FRUIT]
    self.utensil_lost = self.utensil_lost or any(self._pos(u)[2] < OFF_TABLE_Z for u in UTENSILS)

  def score(self) -> dict:
    """1/3 for each fruit resting in the green tray at the end; 0 if the arms ever touched or a utensil ended up off the
    table, so that wrecking the other arm's job can never score.  (After a touch the trial is over at 0.)"""
    if self.touched:
      return {"score": 0.0, "fruit_in_tray": None, "arms_touched": True, "utensil_off_table": None, "steps": self.steps}
    value = 0.0 if self.utensil_lost else sum(self.in_tray) / 3.0
    return {"score": round(value, 3), "fruit_in_tray": sum(self.in_tray), "arms_touched": False,
            "utensil_off_table": self.utensil_lost, "steps": self.steps}
