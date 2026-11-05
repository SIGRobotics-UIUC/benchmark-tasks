"""The scene: Clutter in the middle of the table, three red blocks on the second arm's side of it and four white scraps on the controlled arm's; a green patch painted on the controlled arm's side and the second arm's bin on its own, and what the arms sense and do."""

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
CLUTTER_XY = (0.34, -0.015)
"""The middle of the clutter: red blocks on the worker's side of it, rubbish on the helper's."""
CLUTTER_INNER = (0.34, 0.17)
SWEEP_ZONE_XY = (0.34, 0.24)
"""Where the helper sweeps the rubbish to: a patch on its own side, painted on the table."""
SWEEP_ZONE_INNER = (0.36, 0.14)
SWEEP_BIN_XY = (0.33, -0.30)
"""The worker's bin, on its side."""
SWEEP_REDS = {"red_1": (0.24, -0.07), "red_2": (0.34, -0.06), "red_3": (0.44, -0.07)}
SWEEP_JUNK = {"scrap_1": (0.22, 0.04), "scrap_2": (0.30, 0.03), "scrap_3": (0.38, 0.045), "scrap_4": (0.46, 0.035)}
"""Where each lies to start with: the rubbish each in a lane of its own across the table, so it can be pushed straight
to the helper's side without running into anything."""
SWEEP_START_S = 5.0
SWEEP_PERIOD_S = 9.0
"""The worker starts on a red block at these times, on the dot."""


def sweep_xyz(name: str) -> np.ndarray:
  kind = "red" if name in SWEEP_REDS else "paper"
  x, y = {**SWEEP_REDS, **SWEEP_JUNK}[name]
  return np.array([x, y, Z + props.part_half_height(kind) + 0.001])


def sweep_bin_slot(k: int) -> np.ndarray:
  return np.array([SWEEP_BIN_XY[0], SWEEP_BIN_XY[1] - 0.05 + 0.05 * k, Z + 0.004 + 0.015 + 0.012])


def make_sweep_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
  entities = make_rig(READY_JOINTS)
  entities["sweep_zone"] = _place(lambda: props.zone_spec("sweep_zone", SWEEP_ZONE_INNER, (0.35, 0.6, 0.35, 1.0)), *SWEEP_ZONE_XY)
  entities["bin"] = _place(lambda: props.tray_spec("bin", BOX_INNER, (0.55, 0.42, 0.28, 1.0)), *SWEEP_BIN_XY)
  names = tuple(SWEEP_REDS) + tuple(SWEEP_JUNK)
  for name in names:
    p = sweep_xyz(name)
    kind = "red" if name in SWEEP_REDS else "paper"
    entities[name] = _place(lambda name=name, kind=kind: props.part_spec(name, kind), p[0], p[1], p[2] - Z)
  return _make_cfg(entities, names, {}, 60.0, num_envs, nconmax=400, njmax=2000)
