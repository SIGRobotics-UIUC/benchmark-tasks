"""The scene: one measuring cup of 16 beads in the middle of the table, a bowl in front of each arm, and what the
arms sense and do."""

from __future__ import annotations

import math

import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from sim import props
from sim.env_cfg import DECIMATION, PHYSICS_DT, READY_JOINTS, arm_actions
from sim.robot import FINGER_MAX, GRASP_SITE
from sim.scene import LEFT_ARM, RIGHT_ARM, TABLE_CENTER_X, TABLE_TOP, make_rig

Z = TABLE_TOP
ARMS = (LEFT_ARM, RIGHT_ARM)


def _pose(x: float, y: float, z: float) -> EntityCfg.InitialStateCfg:
  return EntityCfg.InitialStateCfg(pos=(x, y, Z + z))


def _place(spec_fn, x: float, y: float, z: float = 0.0) -> EntityCfg:
  return EntityCfg(spec_fn=spec_fn, init_state=_pose(x, y, z))


def _arms_touching(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The rule of the table: the first touch between the arms ends the episode."""
  return env.scene["arm_contact"].data.found[:, 0] > 0


def _env_pos(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  return env.scene[name].data.root_link_pos_w - env.scene.env_origins


def _site_pos(env: ManagerBasedRlEnv, arm: str) -> torch.Tensor:
  robot = env.scene[arm]
  return robot.data.site_pos_w[:, robot.site_names.index(GRASP_SITE)] - env.scene.env_origins


def _object_pose(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  data = env.scene[name].data
  return torch.cat([data.root_link_pos_w - env.scene.env_origins, data.root_link_quat_w], dim=-1)


def _policy_group(arm: str, objects: tuple[str, ...], extra: dict[str, ObservationTermCfg] | None = None) -> ObservationGroupCfg:
  other = RIGHT_ARM if arm == LEFT_ARM else LEFT_ARM
  asset = {"asset_cfg": SceneEntityCfg(arm)}
  terms: dict[str, ObservationTermCfg] = {
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel, params=asset, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, params=asset, noise=Unoise(n_min=-0.5, n_max=0.5)),
    "last_arm_action": ObservationTermCfg(func=envs_mdp.last_action, params={"action_name": f"{arm}_joints"}),
    "last_grip_action": ObservationTermCfg(func=envs_mdp.last_action, params={"action_name": f"{arm}_gripper"}),
    "other_gripper": ObservationTermCfg(func=_site_pos, params={"arm": other}),
  }
  for name in objects:
    terms[name] = ObservationTermCfg(func=_object_pose, params={"name": name})
  return ObservationGroupCfg({**terms, **(extra or {})}, enable_corruption=True)


def _make_cfg(
  entities: dict[str, EntityCfg],
  objects: tuple[str, ...],
  events: dict[str, EventTermCfg],
  episode_length_s: float,
  num_envs: int,
  nconmax: int,
  njmax: int,
) -> ManagerBasedRlEnvCfg:
  left, right = _policy_group(LEFT_ARM, objects), _policy_group(RIGHT_ARM, objects)
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities=entities,
      sensors=(
        ContactSensorCfg(
          name="arm_contact",
          primary=ContactMatch(mode="subtree", pattern="arm", entity=LEFT_ARM),
          secondary=ContactMatch(mode="subtree", pattern="arm", entity=RIGHT_ARM),
          fields=("found", "force"),
          reduce="netforce",
        ),
      ),
      num_envs=num_envs,
      env_spacing=3.0,
    ),
    observations={"left": left, "right": right},
    actions={**arm_actions(LEFT_ARM), **arm_actions(RIGHT_ARM)},
    events={
      "reset_scene_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
      **events,
    },
    rewards={},
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
      "arm_collision": TerminationTermCfg(func=_arms_touching),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.28, 0.0, TABLE_TOP),
      distance=1.5,
      elevation=-48.0,
      azimuth=180.0,
    ),
    sim=SimulationCfg(
      nconmax=nconmax,
      njmax=njmax,
      mujoco=MujocoCfg(timestep=PHYSICS_DT, iterations=10, ls_iterations=20, impratio=10, cone="elliptic"),
    ),
    decimation=DECIMATION,
    episode_length_s=episode_length_s,
  )
BEADS = tuple(f"bead_{i}" for i in range(16))
MCUP_XY = (0.26, 0.0)
"""Where the one cup of beads stands, on the centre line."""
MBOWL_XY = {LEFT_ARM: (0.33, 0.22), RIGHT_ARM: (0.33, -0.22)}
MBOWLS = {LEFT_ARM: "left_bowl", RIGHT_ARM: "right_bowl"}
MBOWL_INNER = 0.10
MBOWL_HEIGHT = 0.03
SHARE = (6, 10)
"""Beads in its own bowl that make an arm's share: at least 6 of the 16, and no more than 10, or the other is short."""


def make_measuring_cup_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
  entities = make_rig(READY_JOINTS)
  entities["cup"] = _place(props.mcup_spec, *MCUP_XY, 0.001)
  for arm, name in MBOWLS.items():
    entities[name] = _place(lambda name=name: props.bowl_spec(MBOWL_INNER, MBOWL_HEIGHT, name, free=False), *MBOWL_XY[arm])
  spacing = 2.03 * props.BEAD_RADIUS
  for i, bead in enumerate(BEADS):  # One layer of four by four on the cup's floor.
    row, col = i // 4, i % 4
    entities[bead] = _place(
      props.bead_spec, MCUP_XY[0] + (col - 1.5) * spacing, MCUP_XY[1] + (row - 1.5) * spacing,
      0.004 + props.BEAD_RADIUS + 0.002,
    )
  return _make_cfg(entities, ("cup",), {}, 120.0, num_envs, nconmax=800, njmax=3000)
