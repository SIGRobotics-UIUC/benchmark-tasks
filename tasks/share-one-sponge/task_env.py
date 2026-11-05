"""The task: your YAM arm wipes its own plate with the one sponge it shares with a second YAM arm.

    from task_env import TaskEnv
    env = TaskEnv()
    obs = env.reset()
    while not env.done:
      obs = env.step(action)              # 7 numbers: six joint targets (rad), then the gripper, 0 closed to 1 open
    print(env.score())

The controlled arm is the left one (positive y): on the right of the screenshot, which looks toward the arms.  Its plate
is the one in front of it, with five spots of dirt drawn on it.  The second arm (the right one) is scripted
(``sim/partner.py``): it reads the scene's true state, waits for the sponge to lie where it started with the controlled
arm's hand away from it, takes it, wipes its own plate and puts it back, the same way in every trial.  Control is at
20 Hz, ``RUN_LIMIT`` steps at most.  Observations: ``joints`` (13: six joint angles, the gripper opening 0 to 1, six
joint speeds), and two 240 by 320 colour images, ``overhead`` and ``wrist``.
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse

from cameras import Cameras
from sim import props
from sim.benchmark import PLATE_XY, SPONGE_XY, Z, make_sponge_cfg
from sim.env_cfg import ACTION_SCALE, READY_JOINTS
from sim.ik import ArmIk
from sim.partner import Wiper, pos, site
from sim.robot import ARM_JOINTS, FINGER_MAX, GRASP_SITE
from sim.scene import LEFT_ARM, RIGHT_ARM

CONTROL_HZ = 20
RUN_LIMIT = 3600
"""Control steps: 3 minutes at 20 Hz."""
READY = np.array([READY_JOINTS[j] for j in ARM_JOINTS])
SPONGE_REACH = 0.04
"""An arm holds the sponge when its grasp site is this close to the sponge's middle and its fingers are closing."""
WIPE_PRESS = 0.006
"""A spot is wiped when it is under the sponge's face, the sponge's underside within this of the plate's top, the
sponge held."""
SPONGE_HOME = 0.06
"""The sponge is back when it lies on the table, held by nobody, its middle this close to where it started."""
SETTLED_S = 1.0
"""The trial ends once the controlled arm's plate is clean and the sponge back, for this long."""
PARTNER_SEED = 0
"""Seeds the second arm's random waits: fixed, so that it moves the same way in every trial."""


def _row(action) -> np.ndarray:
  return np.r_[(np.asarray(action[:6]) - READY) / ACTION_SCALE, np.clip(action[6], 0.0, 1.0)]


def holding(env: ManagerBasedRlEnv, arm: str) -> bool:
  robot = env.scene[arm]
  fingers = float(robot.data.joint_pos[0, robot.joint_names.index("joint7")])
  return bool(np.linalg.norm(site(env, arm, GRASP_SITE, 0) - pos(env, "sponge", 0)) < SPONGE_REACH and fingers < 0.8 * FINGER_MAX)


def under_sponge(env: ManagerBasedRlEnv, plate_arm: str) -> np.ndarray:
  """Per spot on ``plate_arm``'s plate, whether it is under the sponge's face, the sponge pressed on the plate."""
  s = pos(env, "sponge", 0)
  if s[2] - props.SPONGE[2] / 2 > Z + props.PLATE_HEIGHT + WIPE_PRESS:
    return np.zeros(len(props.DIRT), dtype=bool)
  quat = env.scene["sponge"].data.root_link_quat_w[0:1]
  spots = np.array(PLATE_XY[plate_arm]) + np.array(props.DIRT)
  rel = torch.tensor(np.c_[spots - s[:2], np.zeros(len(spots))], dtype=torch.float32, device=quat.device)
  local = quat_apply_inverse(quat.expand(len(spots), 4), rel).cpu().numpy()
  return (np.abs(local[:, :2]) < np.array(props.SPONGE[:2]) / 2).all(axis=1)


def sponge_home(env: ManagerBasedRlEnv) -> bool:
  p = pos(env, "sponge", 0)
  return bool(np.linalg.norm(p[:2] - np.array(SPONGE_XY)) < SPONGE_HOME and p[2] < Z + props.SPONGE[2]
              and not holding(env, LEFT_ARM) and not holding(env, RIGHT_ARM))


class TaskEnv:
  def __init__(self, device: str | None = None) -> None:
    cfg = make_sponge_cfg()
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
    self.steps = self.ticks = self.clean_ticks = 0
    self.done = self.touched = self.wiped_other = False
    self.wiped = np.zeros(len(props.DIRT), dtype=bool)
    self.partner = Wiper(RIGHT_ARM, self.ik, np.random.default_rng(PARTNER_SEED), self.env.step_dt)
    return self._obs()

  def _account(self) -> None:
    """Spots on its own plate wiped while the controlled arm holds the sponge; a spot on the other plate wiped while it
    holds the sponge is a foul."""
    if holding(self.env, LEFT_ARM):
      self.wiped |= under_sponge(self.env, LEFT_ARM)
      self.wiped_other |= bool(under_sponge(self.env, RIGHT_ARM).any())

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
      self.clean_ticks = self.clean_ticks + 1 if self.wiped.all() and sponge_home(self.env) else 0
      self.done = self.touched or self.wiped_other or self.clean_ticks >= SETTLED_S * self.ticks_per_s
    self.done = self.done or self.steps >= RUN_LIMIT
    return self._obs()

  def score(self) -> dict:
    """The share of the five spots on the controlled arm's plate it wiped, holding the sponge pressed on the plate;
    halved if the sponge is not back where it started at the end (the other arm needs it); 0 if it wiped the other
    plate, or if the arms ever touched."""
    progress = float(self.wiped.mean())
    home = sponge_home(self.env)
    value = 0.0 if self.touched or self.wiped_other else progress * (1.0 if home else 0.5)
    return {"score": round(value, 3), "spots_wiped": int(self.wiped.sum()), "sponge_back": home,
            "wiped_other_plate": self.wiped_other, "arms_touched": self.touched, "steps": self.steps}
