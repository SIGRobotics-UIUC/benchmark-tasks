"""The benchmark's scenes: what is on the table, what the policies see and how the arms act.

Goals and scoring are in ``sim.scoring``, which an agent in privileged mode does not get.
"""

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


# --------------------------------------------------------------------------- #
# Task 1: one bin, two jobs.
# --------------------------------------------------------------------------- #

BIN_XY = (0.26, 0.0)
BIN_INNER = (0.33, 0.33)
BIN_HEIGHT = 0.06
TRAY_INNER = (0.20, 0.16)
TRAY_XY = {LEFT_ARM: (0.26, 0.34), RIGHT_ARM: (0.26, -0.34)}
FRUIT = ("apple", "orange", "banana")
UTENSILS = ("fork", "spoon", "knife")
_SLOTS = [(x, y) for x in (-0.11, 0.0, 0.11) for y in (-0.085, 0.085)]
"""Where the six objects start, in the bin's frame, one to a slot, so that no two start overlapping."""


def _shuffle_into_bin(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Deal the objects into the bin's slots at random, each a little off centre and turned about z."""
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
  events = {"shuffle_into_bin": EventTermCfg(func=_shuffle_into_bin, mode="reset")}
  return _make_cfg(entities, FRUIT + UTENSILS, events, 90.0, num_envs, nconmax=250, njmax=1000)


# --------------------------------------------------------------------------- #
# Task 2: one marker, two drawings.
# --------------------------------------------------------------------------- #

BOARD_SIZE = (0.26, 0.24)
BOARD_XY = {LEFT_ARM: (0.25, 0.31), RIGHT_ARM: (0.25, -0.31)}
STAND_XY = (0.25, 0.0)
STAND_HEIGHT = 0.05
TARGET_RADIUS = 0.08
TARGET_SQUARE = 0.15
def _circle() -> list[tuple[float, float]]:
  return [(TARGET_RADIUS * math.cos(a), TARGET_RADIUS * math.sin(a)) for a in (2 * math.pi * i / 36 for i in range(36))]


def _square() -> list[tuple[float, float]]:
  h = TARGET_SQUARE / 2
  return [(h, h), (-h, h), (-h, -h), (h, -h)]


def make_shared_marker_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
  entities = make_rig(READY_JOINTS)
  entities["stand"] = _place(lambda: props.stand_spec(0.028, STAND_HEIGHT), *STAND_XY)
  entities["left_board"] = _place(lambda: props.board_spec("left_board", BOARD_SIZE, _circle()), *BOARD_XY[LEFT_ARM])
  entities["right_board"] = _place(lambda: props.board_spec("right_board", BOARD_SIZE, _square()), *BOARD_XY[RIGHT_ARM])
  entities["marker"] = _place(props.marker_spec, *STAND_XY, 0.004 + props.MARKER_LENGTH / 2)
  return _make_cfg(entities, ("marker",), {}, 120.0, num_envs, nconmax=100, njmax=600)


# --------------------------------------------------------------------------- #
# Task 3: pouring on a table the other arm is shaking.
# --------------------------------------------------------------------------- #

PATCH_SIZE = (0.26, 0.20)
PATCH_XY = (0.30, 0.28)
SANDER_XY = (0.09, 0.30)
CUP_XY = (0.16, -0.30)
BOWL_XY = (0.30, -0.14)
BOWL_INNER = 0.12
BOWL_HEIGHT = 0.04
BEADS = tuple(f"bead_{i}" for i in range(16))
SHAKE_HZ = 12.0
SHAKE_AMPLITUDE = 0.002
"""Metres, along x and z.  The rig's arm bases are bolted to the world, so the shaking reaches the table top and what
stands on it, not the arms."""
GRIP_REACH = 0.05
"""The gripper has the sander when its grasp site is this close to the handle and its fingers are closing on it."""


def _sander_running(env: ManagerBasedRlEnv) -> torch.Tensor:
  sander = _env_pos(env, "sander")
  patch = torch.tensor(PATCH_XY, device=env.device)
  half = torch.tensor(PATCH_SIZE, device=env.device) / 2
  on_patch = ((sander[:, :2] - patch).abs() < half).all(dim=-1) & (sander[:, 2] < Z + 0.01)
  robot = env.scene[LEFT_ARM]
  fingers = robot.data.joint_pos[:, robot.joint_names.index("joint7")]
  handle = sander + torch.tensor([0.0, 0.0, 0.06], device=env.device)
  held = ((_site_pos(env, LEFT_ARM) - handle).norm(dim=-1) < GRIP_REACH) & (fingers < 0.8 * FINGER_MAX)
  return on_patch & held


def _shake_table(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Runs every step: the shaking lives in the table motor's target."""
  table = env.scene["table"]
  phase = 2.0 * math.pi * SHAKE_HZ * env.episode_length_buf.float() * env.step_dt
  wave = SHAKE_AMPLITUDE * torch.sin(phase) * _sander_running(env).float()
  table.set_joint_position_target(torch.stack([wave, wave], dim=-1))


def _is_sanding(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _sander_running(env).float()[:, None]


def _table_motion(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["table"].data.joint_pos.clone()


def make_shaken_pour_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
  entities = make_rig(READY_JOINTS, vibrating_table=True)
  entities["patch"] = _place(lambda: props.patch_spec(PATCH_SIZE), *PATCH_XY)
  entities["sander"] = _place(props.sander_spec, *SANDER_XY, 0.001)
  entities["cup"] = _place(props.cup_spec, *CUP_XY, 0.001)
  entities["bowl"] = _place(lambda: props.bowl_spec(BOWL_INNER, BOWL_HEIGHT), *BOWL_XY, 0.001)
  spacing = 2.05 * props.BEAD_RADIUS
  for i, bead in enumerate(BEADS):
    layer, row, col = i // 4, (i % 4) // 2, i % 2
    entities[bead] = _place(
      props.bead_spec,
      CUP_XY[0] + (col - 0.5) * 2 * spacing,
      CUP_XY[1] + (row - 0.5) * 2 * spacing,
      props.CUP_FLOOR + props.BEAD_RADIUS + 0.002 + layer * 2 * spacing,
    )
  events = {"shake_table": EventTermCfg(func=_shake_table, mode="step")}
  cfg = _make_cfg(entities, ("sander", "cup", "bowl"), events, 90.0, num_envs, nconmax=800, njmax=3000)
  # The pouring arm sees the shaking only through the table and the sanding arm through its own sander.
  cfg.observations["right"].terms["table_motion"] = ObservationTermCfg(func=_table_motion)
  cfg.observations["left"].terms["sanding"] = ObservationTermCfg(func=_is_sanding)
  return cfg
