# Fill your own kit from a parts tray another robot arm is also picking from

One YAM arm, the one under test, and a second YAM arm each fill a kit tray beside them from one grey tray of mixed
parts between them. The tested arm's kit is 2 red cubes, 1 blue cylinder and 1 yellow block, into its green tray; the
second arm's is 1 red cube, 2 blue cylinders and 1 yellow block, into its blue tray. The grey tray holds exactly the
eight parts the two kits need, so a part taken beyond one's own kit leaves the other kit short. The second arm keeps
out of the grey tray while the tested arm's gripper is down in it, and backs off if it comes down while it is on its
way in. The arms must never touch.

## Running a trial

Python 3.11 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is used when present.

    uv sync
    MUJOCO_GL=egl uv run python run_trial.py --policy hold                 # the arm stays still: 0.0
    MUJOCO_GL=egl uv run python run_trial.py --policy demo --video demo.mp4 # the demonstration: 1.0
    MUJOCO_GL=egl uv run python run_trial.py --policy mymodule:MyPolicy     # your policy
    MUJOCO_GL=egl uv run python run_trial.py --screenshot setup.png         # the starting layout

Each prints the score as JSON, for example
`{"score": 1.0, "kit": ["blue_1", "red_1", "red_2", "yellow_1"], "arms_touched": false, "other_kit_progress": 1.0, "steps": 1143}`.

**Runtime:** about 1 to 4 minutes per trial on an RTX 2070 Super, most of it rendering the two cameras every step.

**Seeds:** none to choose. The layout is fixed, and the second arm's random waits are seeded (`PARTNER_SEED = 0` in
`task_env.py`), so it moves the same way in every trial.

## Interface

`task_env.TaskEnv`: `reset()` and `step(action)` return the observation; `done` says when the trial is over; `score()`
gives the result.

- **Control:** 20 Hz, at most 3,600 steps (3 minutes). Inside each step the simulation steps at 50 Hz, the second arm
  with it; the arm under test holds its command for the step.
- **Action:** 7 numbers: six joint angle targets in radians, then the gripper, from 0 (closed) to 1 (open).
- **Observation:** `joints`, 13 floats (six joint angles, the gripper opening from 0 to 1, six joint speeds); `overhead`
  and `wrist`, 240 by 320 RGB images from a camera above the table and one on the controlled arm's wrist.
- **The controlled arm:** the left arm in the scene (positive y), which is the arm on the right of the screenshot, as it
  looks toward the arms. Its kit tray is the green one beside it.

## Scoring

`TaskEnv.score()`:

- The share of the kit resting inside the green tray at the end (inside its walls, below 6 cm over the table), counting
  no kind beyond what the kit asks: each red cube up to two, the blue cylinder and the yellow block one each, out of 4.
- 0 if the green tray holds more of any kind than the kit asks: the grey tray holds no spares, so the other kit is then
  short.
- 0 if the arms touched at any point. The trial stops at the first touch, and once the kit has been complete for 1 s.

A part knocked out of the grey tray still counts once it is in the green tray.

## The demo

`--policy demo` drives the controlled arm with the same script as the second arm (`Kitter` in `sim/partner.py`), at
20 Hz. Unlike a policy, it reads the simulator's true state (where the parts and the other arm are), not the cameras:
it shows that the task can be done with this arm in this scene. It waits a random while; then, part by part, it waits
until the other gripper is out of the grey tray, comes down over the part its kit needs that lies nearest its own side,
picks it across its narrower side, carries it high to the next free corner of its kit tray and lets go. With both arms
scripted it filled both kits, without the arms touching, in each of 8 runs with different random waits.

## Running Dream-1

`dream_run.py` runs Dream-1 (`dreamlabs/dream-1-beta`) on the task the way Dream Labs' own YAM MuJoCo example drives a
simulator: both arms' joints and gripper openings, and nine-frame windows from the YAM station's three calibrated
cameras (top, left wrist, right wrist), in the station's request keys. Dream-1 controls the left arm; its right-arm
rows are not executed, since that arm is the task's scripted second arm. Simulated time stands still while a request is
out.

    uv sync --extra dream
    MUJOCO_GL=egl uv run python dream_run.py --key-file <file holding the API key> --out dream_run
    MUJOCO_GL=egl uv run python dream_run.py --dry-run                     # checks the request; no API call

Our run scored 0.0: in 30 s (25 requests of 24 rows each) Dream-1 reached over the grey tray and its green tray without picking anything up, and its arm then touched the other arm, which had 3 of its 4 parts in place; the trial stopped there.

## Third-party assets and licenses

- `assets/yam_pro_linear_4310/`: the YAM Pro arm and linear_4310 gripper model (MJCF and STL meshes), from I2RT
  Robotics' i2rt (https://github.com/i2rt-robotics/i2rt), copyright (c) I2RT Robotics, MIT License. The license text
  and copyright notice are in `assets/yam_pro_linear_4310/LICENSE`; `assets/yam_pro_linear_4310/README.md` says what
  was changed.
- No other third-party files are included: everything else in this folder was written for this task, though some
  numbers in it are taken from published sources: the arm's joint gains and limits from i2rt, and the YAM station's
  camera calibration and request format from Dream Labs' SDK example (`cameras.py`, `dream_run.py`). The Python
  packages it runs on (mjlab, MuJoCo, PyTorch, mediapy, Pillow and, for `dream_run.py` only, dreamlabs-client) are
  installed from their own sources by `uv sync`, under their own licenses.

## Files

- `task_env.py`: the task, the second arm in it, and the scoring.
- `run_trial.py`: runs and scores one trial; also the demo, the screenshot and videos.
- `dream_run.py`: runs Dream-1 on the task (needs `uv sync --extra dream` and an API key).
- `cameras.py`: the policy's two cameras, the YAM station's three cameras for Dream-1, and the wide view used for the
  screenshot and videos (not given to the policy).
- `sim/`: the scene (`benchmark.py`, `props.py`), the YAM model with its controller (`robot.py`, `env_cfg.py`), the
  table (`scene.py`), inverse kinematics (`ik.py`), and the arms' script (`partner.py`).
- `assets/`: the YAM Pro arm and linear_4310 gripper model, from i2rt (MIT License; see above).
