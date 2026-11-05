"""The scene: red blocks and clutter scattered over the table, the controlled arm's bin in its back corner, Arm 1's hand
broom on Arm 1's side, and what the arms sense and do.  The layout of sweep-while-picking, mirrored left to right."""

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

BIN_XY = (0.16, 0.34)
"""The controlled arm's bin, in its back corner of the table."""
BROOM_XY = (0.22, -0.36)
"""Arm 1's broom, standing on its bristles, its length along x, on Arm 1's side."""
RED_BLOCKS = ((0.20, -0.08), (0.32, -0.08), (0.40, 0.00), (0.15, -0.02))
"""The four red blocks, all inside the area Arm 1 sweeps."""
CLUTTER = (
  ("ball", 0.15, -0.21), ("paper", 0.18, -0.27), ("box", 0.26, -0.24),
  ("ball", 0.36, -0.20), ("box", 0.44, -0.12), ("cube", 0.42, -0.34),
)
"""The six clutter items, (kind, x, y): two in each 20 cm band the broom covers, so no single stroke clears them."""
CLUTTER_NAMES = tuple(f"clutter_{i}" for i in range(len(CLUTTER)))
BLOCK_NAMES = tuple(f"red_block_{i}" for i in range(len(RED_BLOCKS)))


def _contact(name: str, primary: tuple[str, str], secondary: tuple[str, str]) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="subtree", pattern=primary[1], entity=primary[0]),
    secondary=ContactMatch(mode="subtree", pattern=secondary[1], entity=secondary[0]),
    fields=("found",),
    reduce="netforce",
  )


def make_pick_cfg(picker: str = LEFT_ARM, num_envs: int = 1) -> ManagerBasedRlEnvCfg:
  sweeper = RIGHT_ARM if picker == LEFT_ARM else LEFT_ARM
  entities = make_rig(READY_JOINTS)
  entities["bin"] = _place(props.bin_spec, *BIN_XY)
  entities["broom"] = _place(props.broom_spec, *BROOM_XY, 0.001)
  for name, (x, y) in zip(BLOCK_NAMES, RED_BLOCKS):
    entities[name] = _place(props.red_block_spec, x, y, props.BLOCK / 2 + 0.001)
  for name, (kind, x, y) in zip(CLUTTER_NAMES, CLUTTER):
    entities[name] = _place(lambda kind=kind: props.clutter_spec(kind), x, y, props.clutter_half_height(kind) + 0.001)
  cfg = _make_cfg(entities, ("broom",), {}, 200.0, num_envs, nconmax=600, njmax=3000)
  sensors = [_contact("broom_picker_contact", ("broom", "broom"), (picker, "arm"))]
  # A red block in the controlled arm's hand counts as part of it: touching Arm 1 or its broom with it is contact.
  for name in BLOCK_NAMES:
    sensors.append(_contact(f"{name}_sweeper_contact", (name, "red_block"), (sweeper, "arm")))
    sensors.append(_contact(f"{name}_broom_contact", (name, "red_block"), ("broom", "broom")))
  cfg.scene.sensors = (*cfg.scene.sensors, *sensors)
  return cfg
