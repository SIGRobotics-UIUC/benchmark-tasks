"""The two-arm rig as an mjlab env: the starting point for a task.

The scene is ``sim.scene``'s.  Each arm is ``sim.robot``'s, under the same controller.  The action is, per arm, six
joint position offsets from ``READY_JOINTS`` and a gripper command, 0 closed to 1 open: left arm then right, 14 numbers
(``ACTION_TERMS``).  The policy sees both arms, the cube and its last action.  Physics steps at 5 ms, four to a 50 Hz
policy step.

The task itself is a placeholder: stay up, move smoothly, end at the time limit.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from sim.robot import ARM_JOINTS, FINGER_JOINTS, FINGER_MAX
from sim.scene import LEFT_ARM, RIGHT_ARM, TABLE_CENTER_X, TABLE_TOP, make_entities

READY_JOINTS = {"joint1": 0.0, "joint2": 0.63, "joint3": 0.74, "joint4": -0.89, "joint5": 0.0, "joint6": 0.0}
"""Gripper 0.22 m out and 0.18 m above the table, pointing 45 deg down, well clear of the other arm."""
ACTION_SCALE = 0.5
"""rad per unit of action, about ``READY_JOINTS``."""
PHYSICS_DT = 0.005
DECIMATION = 4
"""5 ms physics, 50 Hz policy."""
ACTION_TERMS = (f"{LEFT_ARM}_joints", f"{LEFT_ARM}_gripper", f"{RIGHT_ARM}_joints", f"{RIGHT_ARM}_gripper")


def _cube_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The cube in the env's own frame, so every env sees the same table."""
  return env.scene["cube"].data.root_link_pos_w - env.scene.env_origins


def arm_actions(arm: str) -> dict[str, JointPositionActionCfg]:
  return {
    f"{arm}_joints": JointPositionActionCfg(
      entity_name=arm, actuator_names=ARM_JOINTS, scale=ACTION_SCALE, use_default_offset=True, preserve_order=True,
    ),
    f"{arm}_gripper": JointPositionActionCfg(
      entity_name=arm, actuator_names=(FINGER_JOINTS[0],), scale=FINGER_MAX, use_default_offset=False,
    ),
  }


def make_env_cfg(num_envs: int = 1, episode_length_s: float = 10.0) -> ManagerBasedRlEnvCfg:
  policy: dict[str, ObservationTermCfg] = {}
  for arm in (LEFT_ARM, RIGHT_ARM):
    asset = {"asset_cfg": SceneEntityCfg(arm)}
    policy[f"{arm}_joint_pos"] = ObservationTermCfg(
      func=envs_mdp.joint_pos_rel, params=asset, noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    policy[f"{arm}_joint_vel"] = ObservationTermCfg(
      func=envs_mdp.joint_vel_rel, params=asset, noise=Unoise(n_min=-0.5, n_max=0.5),
    )
  policy["cube_pos"] = ObservationTermCfg(func=_cube_pos)
  policy["actions"] = ObservationTermCfg(func=envs_mdp.last_action)

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities=make_entities(READY_JOINTS),
      num_envs=num_envs,
      env_spacing=3.0,
    ),
    observations={
      "actor": ObservationGroupCfg(policy, enable_corruption=True),
      "critic": ObservationGroupCfg({**policy}, enable_corruption=False),
    },
    actions={**arm_actions(LEFT_ARM), **arm_actions(RIGHT_ARM)},
    events={
      "reset_scene_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
    },
    rewards={
      "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=1.0),
      "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
    },
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(TABLE_CENTER_X, 0.0, TABLE_TOP),
      distance=2.4,
      elevation=-28.0,
      azimuth=205.0,
    ),
    sim=SimulationCfg(
      nconmax=60,
      njmax=400,
      mujoco=MujocoCfg(timestep=PHYSICS_DT, iterations=10, ls_iterations=20, impratio=10, cone="elliptic"),
    ),
    decimation=DECIMATION,
    episode_length_s=episode_length_s,
  )
