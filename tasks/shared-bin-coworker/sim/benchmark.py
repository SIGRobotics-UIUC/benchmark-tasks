"""The scene: a shared bin of fruit and utensils between two YAM arms, a tray for each, and what the arms sense and do."""

from __future__ import annotations

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
from sim.robot import GRASP_SITE
from sim.scene import LEFT_ARM, RIGHT_ARM, TABLE_TOP, make_rig

Z = TABLE_TOP

BIN_XY = (0.26, 0.0)
BIN_INNER = (0.33, 0.33)
BIN_HEIGHT = 0.06
TRAY_INNER = (0.20, 0.16)
TRAY_XY = {LEFT_ARM: (0.26, 0.34), RIGHT_ARM: (0.26, -0.34)}
FRUIT = ("apple", "orange", "banana")
UTENSILS = ("fork", "spoon", "knife")
_SLOTS = [(x, y) for x in (-0.11, 0.0, 0.11) for y in (-0.085, 0.085)]
"""Where the six objects start, in the bin's frame, one to a slot, so that no two start overlapping."""


def _place(spec_fn, x: float, y: float, z: float = 0.0) -> EntityCfg:
  return EntityCfg(spec_fn=spec_fn, init_state=EntityCfg.InitialStateCfg(pos=(x, y, Z + z)))


def _arms_touching(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The first touch between the arms ends the episode."""
  return env.scene["arm_contact"].data.found[:, 0] > 0


def _site_pos(env: ManagerBasedRlEnv, arm: str) -> torch.Tensor:
  robot = env.scene[arm]
  return robot.data.site_pos_w[:, robot.site_names.index(GRASP_SITE)] - env.scene.env_origins


def _object_pose(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  data = env.scene[name].data
  return torch.cat([data.root_link_pos_w - env.scene.env_origins, data.root_link_quat_w], dim=-1)


def _policy_group(arm: str, objects: tuple[str, ...]) -> ObservationGroupCfg:
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
  return ObservationGroupCfg(terms, enable_corruption=True)


def _shuffle_into_bin(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Deal the objects into the bin's slots, each a little off centre and turned about z; a seeded reset deals the same
  layout every time."""
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  order = torch.rand(len(ids), len(_SLOTS), device=env.device).argsort(dim=1)
  slots = torch.tensor(_SLOTS, device=env.device)
  for i, name in enumerate(FRUIT + UTENSILS):
    entity = env.scene[name]
    state = entity.data.default_root_state[ids].clone()
    xy = slots[order[:, i]] + (torch.rand(len(ids), 2, device=env.device) - 0.5) * 0.02
    state[:, 0] = BIN_XY[0] + xy[:, 0]
    state[:, 1] = BIN_XY[1] + xy[:, 1]
    state[:, :3] += env.scene.env_origins[ids]
    yaw = (torch.rand(len(ids), device=env.device) - 0.5) * 0.6
    state[:, 3] = torch.cos(yaw / 2)
    state[:, 6] = torch.sin(yaw / 2)
    entity.write_root_state_to_sim(state, ids)


def make_shared_bin_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
  entities = make_rig(READY_JOINTS)
  entities["bin"] = _place(lambda: props.bin_spec(BIN_INNER, BIN_HEIGHT), *BIN_XY)
  entities["fruit_tray"] = _place(lambda: props.tray_spec("fruit_tray", TRAY_INNER, (0.3, 0.6, 0.3, 1.0)), *TRAY_XY[LEFT_ARM])
  entities["utensil_tray"] = _place(lambda: props.tray_spec("utensil_tray", TRAY_INNER, (0.3, 0.4, 0.7, 1.0)), *TRAY_XY[RIGHT_ARM])
  for (x, y), name in zip(_SLOTS, FRUIT + UTENSILS):
    spec_fn = props.fruit_spec if name in FRUIT else props.utensil_spec
    z = props.BIN_FLOOR + (0.038 if name in FRUIT else 0.008)
    entities[name] = _place(lambda spec_fn=spec_fn, name=name: spec_fn(name), BIN_XY[0] + x, BIN_XY[1] + y, z)
  objects = FRUIT + UTENSILS
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
    observations={"left": _policy_group(LEFT_ARM, objects), "right": _policy_group(RIGHT_ARM, objects)},
    actions={**arm_actions(LEFT_ARM), **arm_actions(RIGHT_ARM)},
    events={
      "reset_scene_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
      "shuffle_into_bin": EventTermCfg(func=_shuffle_into_bin, mode="reset"),
    },
    rewards={},
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
      "arm_collision": TerminationTermCfg(func=_arms_touching),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD, lookat=(0.28, 0.0, TABLE_TOP), distance=1.5, elevation=-48.0,
      azimuth=180.0,
    ),
    sim=SimulationCfg(
      nconmax=250,
      njmax=1000,
      mujoco=MujocoCfg(timestep=PHYSICS_DT, iterations=10, ls_iterations=20, impratio=10, cone="elliptic"),
    ),
    decimation=DECIMATION,
    episode_length_s=90.0,
  )
