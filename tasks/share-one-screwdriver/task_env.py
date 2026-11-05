"""The task: your YAM arm drives its own screw with the one screwdriver it shares with a second YAM arm.

    from task_env import TaskEnv
    env = TaskEnv()
    obs = env.reset()
    while not env.done:
      obs = env.step(action)              # 7 numbers: six joint targets (rad), then the gripper, 0 closed to 1 open
    print(env.score())

The controlled arm is the left one (positive y): on the right of the screenshot, which looks toward the arms.  Its screw
is in the block in front of it.  The second arm (the right one) is scripted (``sim/partner.py``): it reads the scene's
true state, waits for the screwdriver to stand in its holder with the controlled arm's hand away from it, takes it,
drives its own screw half a turn and stands it back, the same way in every trial.  Control is at 20 Hz, ``RUN_LIMIT``
steps at most.  Observations: ``joints`` (13: six joint angles, the gripper opening 0 to 1, six joint speeds), and two
240 by 320 colour images, ``overhead`` and ``wrist``.
"""

from __future__ import annotations

import numpy as np
import torch

from cameras import Cameras
from sim import props
from sim.benchmark import DRIVER_XY, SCREWS, Z, driver_rest_z, make_screwdriver_cfg
from sim.env_cfg import ACTION_SCALE, READY_JOINTS
from sim.ik import ArmIk
from sim.partner import Driver, pos, site
from sim.robot import ARM_JOINTS, FINGER_MAX, GRASP_SITE
from sim.scene import LEFT_ARM, RIGHT_ARM

from mjlab.envs import ManagerBasedRlEnv

CONTROL_HZ = 20
RUN_LIMIT = 3600
"""Control steps: 3 minutes at 20 Hz."""
READY = np.array([READY_JOINTS[j] for j in ARM_JOINTS])
DRIVER_REACH = 0.04
"""An arm holds the screwdriver when its grasp site is this close to the handle's middle and its fingers are closing."""
SCREW_MOVED = 0.01
"""Radians: a screw turning less than this in a step does not count against the arm holding the driver."""
SETTLED_S = 1.0
"""The trial ends once the controlled arm's screw is home and the driver back in its holder, for this long."""
PARTNER_SEED = 0
"""Seeds the second arm's random waits: fixed, so that it moves the same way in every trial."""


def _row(action) -> np.ndarray:
  return np.r_[(np.asarray(action[:6]) - READY) / ACTION_SCALE, np.clip(action[6], 0.0, 1.0)]


def holding(env: ManagerBasedRlEnv, arm: str) -> bool:
  robot = env.scene[arm]
  fingers = float(robot.data.joint_pos[0, robot.joint_names.index("joint7")])
  return bool(np.linalg.norm(site(env, arm, GRASP_SITE, 0) - pos(env, "driver", 0)) < DRIVER_REACH and fingers < 0.8 * FINGER_MAX)


def turned(env: ManagerBasedRlEnv, arm: str) -> float:
  """How far ``arm``'s screw has been driven in: clockwise from above, radians."""
  return -float(env.scene[SCREWS[arm]].data.joint_pos[0, 0]) + 0.0  # Never -0.0.


def driver_home(env: ManagerBasedRlEnv) -> bool:
  p = pos(env, "driver", 0)
  return bool(np.linalg.norm(p[:2] - np.array(DRIVER_XY)) < 0.015 and p[2] > Z + driver_rest_z() - 0.01
              and not holding(env, LEFT_ARM) and not holding(env, RIGHT_ARM))


class TaskEnv:
  def __init__(self, device: str | None = None) -> None:
    cfg = make_screwdriver_cfg()
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
    self.done = self.touched = self.wrong_screw = False
    self.last_other = turned(self.env, RIGHT_ARM)
    self.partner = Driver(RIGHT_ARM, self.ik, np.random.default_rng(PARTNER_SEED), self.env.step_dt)
    return self._obs()

  def _account(self) -> None:
    """The other arm's screw turning while the controlled arm holds the driver and the other arm does not is a foul."""
    now = turned(self.env, RIGHT_ARM)
    if abs(now - self.last_other) > SCREW_MOVED and holding(self.env, LEFT_ARM) and not holding(self.env, RIGHT_ARM):
      self.wrong_screw = True
    self.last_other = now

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
      home = turned(self.env, LEFT_ARM) >= props.SCREW_TURN and driver_home(self.env)
      self.home_ticks = self.home_ticks + 1 if home else 0
      self.done = self.touched or self.wrong_screw or self.home_ticks >= SETTLED_S * self.ticks_per_s
    self.done = self.done or self.steps >= RUN_LIMIT
    return self._obs()

  def score(self) -> dict:
    """How far the controlled arm's screw has been driven in, over half a turn, up to 1; halved if the driver is not
    back in its holder at the end (the other arm needs it); 0 if the controlled arm turned the other screw, or if the
    arms ever touched."""
    progress = min(max(turned(self.env, LEFT_ARM), 0.0) / props.SCREW_TURN, 1.0)
    home = driver_home(self.env)
    value = 0.0 if self.touched or self.wrong_screw else progress * (1.0 if home else 0.5)
    return {"score": round(value, 3), "screw_turned_deg": round(float(np.degrees(turned(self.env, LEFT_ARM))), 1),
            "driver_back": home, "turned_other_screw": self.wrong_screw, "arms_touched": self.touched, "steps": self.steps}
