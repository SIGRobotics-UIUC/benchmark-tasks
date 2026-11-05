"""The scene: A small grey tray, station B, just on the second arm's side of the middle; a green mat of red blocks on the controlled arm's side; the second arm's crate on its side, and what the arms sense and do."""

from __future__ import annotations

import math

import numpy as np
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
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
WORKER = RIGHT_ARM
"""The arm with the fixed routine."""
HELPER = LEFT_ARM
"""The arm tested."""
CRATE_XY = (0.33, -0.30)
"""The worker's crate, on its side."""
BOX_INNER = (0.16, 0.18)


def make_resupply_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
  entities = make_rig(READY_JOINTS)
  entities["station_b"] = _place(lambda: props.tray_spec("station_b", STATION_B_INNER, (0.5, 0.5, 0.52, 1.0)), *STATION_B_XY)
  entities["supply"] = _place(lambda: props.desk_mat_spec("supply", (0.35, 0.6, 0.35, 1.0)), *SUPPLY_XY)
  entities["crate"] = _place(lambda: props.tray_spec("crate", BOX_INNER, (0.55, 0.42, 0.28, 1.0)), *CRATE_XY)
  for name in BLOCKS_RESUPPLY:
    p = resupply_start(name)
    entities[name] = _place(lambda name=name: props.part_spec(name, "red"), p[0], p[1], p[2] - Z)
  return _make_cfg(entities, BLOCKS_RESUPPLY, {}, 110.0, num_envs, nconmax=400, njmax=2000)
STATION_B_XY = (0.30, -0.06)
"""Station B, where the worker takes a block from every so often: a small tray just on the worker's side of the middle."""
STATION_B_INNER = (0.07, 0.07)
SUPPLY_XY = (0.31, 0.26)
"""Station A, the helper's supply: a mat of blocks on its own side, 20 cm square."""
SUPPLY_INNER = props.DESK_MAT
BLOCKS_RESUPPLY = tuple(f"block_{i}" for i in range(1, 7))
"""Six blocks: the first on station B to start with, the other five in the supply."""
RESUPPLY_START_S = 4.0
RESUPPLY_PERIOD_S = 11.0
RESUPPLY_VISITS = 8
"""The worker takes from station B at these times, on the dot, every visit: whatever is there.  Eight visits for six
blocks: two to spare, for a block that comes late."""


def resupply_slot(k: int) -> np.ndarray:
  """Where the worker lets go on its ``k``-th visit: two rows of four in the crate."""
  return np.array([CRATE_XY[0] + (-0.04, 0.04)[k % 2], CRATE_XY[1] - 0.06 + 0.04 * (k // 2), Z + 0.004 + 0.015 + 0.012])


def resupply_start(name: str) -> np.ndarray:
  k = BLOCKS_RESUPPLY.index(name)
  half = props.part_half_height("red")
  if k == 0:
    return np.array([STATION_B_XY[0], STATION_B_XY[1], Z + 0.004 + half + 0.001])
  # Like the five on a die, 10 cm apart: room for open fingers between them.
  dx, dy = [(-0.05, -0.05), (0.05, -0.05), (0.0, 0.0), (-0.05, 0.05), (0.05, 0.05)][k - 1]
  return np.array([SUPPLY_XY[0] + dx, SUPPLY_XY[1] + dy, Z + 0.003 + half + 0.001])


def station_b_take() -> np.ndarray:
  """Where the worker reaches for a block on station B: its middle."""
  return np.array([STATION_B_XY[0], STATION_B_XY[1], Z + 0.004 + props.part_half_height("red")])
