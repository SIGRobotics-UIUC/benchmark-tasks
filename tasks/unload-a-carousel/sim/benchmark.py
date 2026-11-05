"""The scene: A motor-driven carousel between the arms that turns an eighth of a turn every 6 s; a green mat of red blocks on the second arm's side and the controlled arm's crate on its own, and what the arms sense and do."""

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
BOX_INNER = (0.16, 0.18)
CAROUSEL_XY = (0.32, 0.0)
INDEX_PERIOD_S = 6.0
INDEX_MOVE_S = 1.5
"""The carousel turns an eighth of a turn, counter-clockwise seen from above, in the first ``INDEX_MOVE_S`` of every
``INDEX_PERIOD_S``, from the start of the trial, and stands still the rest of the time."""
LOAD_ANGLE = -90.0
"""Degrees, from +x: the station on the worker's side, nearest it, that the worker loads."""
CONVEYOR_BLOCKS = tuple(f"block_{i}" for i in range(1, 6))
LOAD_SUPPLY_XY = (0.31, -0.30)
"""The worker's supply: five blocks on a mat on its side."""
CONVEYOR_CRATE_XY = (0.33, 0.30)
"""The helper's crate, on its side."""
LOAD_EVERY = 2
"""The worker loads a block every this many indexes, ..."""
FIRST_LOAD = 1
"""... the first in the stand-still after this index."""
LOAD_LEAD_S = 3.6
"""Seconds before an index starts that the worker sets off for the block it will load after it: it lowers the block
onto the station from 1.9 s into the stand-still, and is away 1.5 s later.  A block not taken off comes round to the
worker's station again eight indexes after it was loaded, when the worker loads another on top of it."""


def carousel_angle(t):
  """The carousel's turn, radians, ``t`` seconds into the trial: works on floats and tensors alike."""
  k = t // INDEX_PERIOD_S
  s = (t - k * INDEX_PERIOD_S) / INDEX_MOVE_S
  s = s.clamp(0.0, 1.0) if torch.is_tensor(s) else min(max(s, 0.0), 1.0)
  return (k + 3 * s ** 2 - 2 * s ** 3) * (math.pi / 4)


def station_xyz(deg: float, kind: str = "red") -> np.ndarray:
  """A part's middle, resting on the carousel's station at ``deg`` degrees from +x."""
  a = math.radians(deg)
  return np.array([CAROUSEL_XY[0] + props.CAROUSEL_STATION_R * math.cos(a),
                   CAROUSEL_XY[1] + props.CAROUSEL_STATION_R * math.sin(a), Z + props.CAROUSEL[1] + props.part_half_height(kind)])


def load_supply_xyz(name: str) -> np.ndarray:
  k = CONVEYOR_BLOCKS.index(name)
  dx, dy = [(-0.05, -0.05), (0.05, -0.05), (0.0, 0.0), (-0.05, 0.05), (0.05, 0.05)][k]
  return np.array([LOAD_SUPPLY_XY[0] + dx, LOAD_SUPPLY_XY[1] + dy, Z + 0.003 + props.part_half_height("red") + 0.001])


def load_times() -> list[float]:
  """When the worker sets off for each block."""
  return [(FIRST_LOAD + LOAD_EVERY * k) * INDEX_PERIOD_S - LOAD_LEAD_S for k in range(len(CONVEYOR_BLOCKS))]


def conveyor_crate_slot(k: int) -> np.ndarray:
  dx, dy = [(-0.04, -0.05), (0.04, -0.05), (-0.04, 0.0), (0.04, 0.0), (-0.04, 0.05), (0.04, 0.05)][k % 6]
  return np.array([CONVEYOR_CRATE_XY[0] + dx, CONVEYOR_CRATE_XY[1] + dy, Z + 0.004 + 0.015 + 0.012])


def _drive_carousel(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Runs every step: the carousel's motor follows its timetable."""
  t = env.episode_length_buf.float() * env.step_dt
  env.scene["carousel"].set_joint_position_target(carousel_angle(t)[:, None])


def _carousel_cfg() -> EntityCfg:
  stiffness = 40.0
  return EntityCfg(
    spec_fn=props.carousel_spec,
    init_state=_pose(*CAROUSEL_XY, 0.0),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(target_names_expr=("carousel_turn",), stiffness=stiffness,
                                   damping=2.0 * math.sqrt(stiffness * 0.02), effort_limit=50.0),
      ),
    ),
  )


def make_conveyor_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
  entities = make_rig(READY_JOINTS)
  entities["carousel"] = _carousel_cfg()
  entities["supply"] = _place(lambda: props.desk_mat_spec("supply", (0.35, 0.6, 0.35, 1.0)), *LOAD_SUPPLY_XY)
  entities["crate"] = _place(lambda: props.tray_spec("crate", BOX_INNER, (0.55, 0.42, 0.28, 1.0)), *CONVEYOR_CRATE_XY)
  for name in CONVEYOR_BLOCKS:
    p = load_supply_xyz(name)
    entities[name] = _place(lambda name=name: props.part_spec(name, "red"), p[0], p[1], p[2] - Z)
  events = {"drive_carousel": EventTermCfg(func=_drive_carousel, mode="step")}
  return _make_cfg(entities, CONVEYOR_BLOCKS, events, 100.0, num_envs, nconmax=400, njmax=2000)
