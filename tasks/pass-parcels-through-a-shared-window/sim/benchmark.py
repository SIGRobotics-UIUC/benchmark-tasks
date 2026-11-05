"""The scene: a wall down the middle of the table with one opening in it, a mat of parcels beside each arm and a
tray for each arm's parcels just beyond the opening on the other side, and what the arms sense and do."""

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
WINDOW_XY = (0.335, 0.0)
"""The wall's middle, which is the window's: the wall runs along x on the centre line, between the arms."""
HATCH_TRAY_XY = {LEFT_ARM: (0.335, -0.10), RIGHT_ARM: (0.335, 0.10)}
"""Where each arm sends its parcels: a tray just through the window, on the other arm's side."""
HATCH_TRAY_INNER = (0.12, 0.10)
PARCEL_MAT_XY = {LEFT_ARM: (0.30, 0.27), RIGHT_ARM: (0.30, -0.27)}
PARCELS = {
  "left_parcel_1": (LEFT_ARM, "red", (-0.05, 0.0)), "left_parcel_2": (LEFT_ARM, "red", (0.05, 0.0)),
  "right_parcel_1": (RIGHT_ARM, "blue", (-0.05, 0.0)), "right_parcel_2": (RIGHT_ARM, "blue", (0.05, 0.0)),
}
"""Each parcel: whose it is, its kind, and where on its owner's mat."""


def make_window_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
  entities = make_rig(READY_JOINTS)
  entities["window_wall"] = _place(props.window_wall_spec, *WINDOW_XY)
  for arm, name in ((LEFT_ARM, "left_tray"), (RIGHT_ARM, "right_tray")):
    colour = (0.35, 0.6, 0.35, 1.0) if arm == LEFT_ARM else (0.35, 0.45, 0.75, 1.0)
    entities[name] = _place(lambda name=name, colour=colour: props.tray_spec(name, HATCH_TRAY_INNER, colour), *HATCH_TRAY_XY[arm])
  for arm, name in ((LEFT_ARM, "left_mat"), (RIGHT_ARM, "right_mat")):
    colour = (0.35, 0.6, 0.35, 1.0) if arm == LEFT_ARM else (0.35, 0.45, 0.75, 1.0)
    entities[name] = _place(lambda name=name, colour=colour: props.desk_mat_spec(name, colour), *PARCEL_MAT_XY[arm])
  for name, (arm, kind, (dx, dy)) in PARCELS.items():
    z = 0.003 + props.part_half_height(kind) + 0.001
    entities[name] = _place(lambda name=name, kind=kind: props.part_spec(name, kind),
                            PARCEL_MAT_XY[arm][0] + dx, PARCEL_MAT_XY[arm][1] + dy, z)
  return _make_cfg(entities, tuple(PARCELS), {}, 90.0, num_envs, nconmax=600, njmax=3000)
