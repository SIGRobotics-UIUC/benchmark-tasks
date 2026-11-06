# Rescue blocks from clutter before another robot arm sweeps it away

Five pieces lie in a row of lanes across the middle of the table: three red blocks on the tested arm's side and two
white scraps on the second arm's. The second YAM arm works to a fixed routine: from 12 s in, every 10 s, it puts its
closed hand down on the tested arm's side of the next lane, nearest first, and pushes everything in the lane across onto
a grey patch on its own side, without waiting or looking. The arm under test must get each red block into its bin before
its lane is swept, and keep its hand out of the sweep. The arms must never touch.

## Running a trial

Python 3.11 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is used when present.

    uv sync
    MUJOCO_GL=egl uv run python run_trial.py --policy hold                 # the arm stays still: 0.0
    MUJOCO_GL=egl uv run python run_trial.py --policy demo --video demo.mp4 # the demonstration: 1.0
    MUJOCO_GL=egl uv run python run_trial.py --policy mymodule:MyPolicy     # your policy
    MUJOCO_GL=egl uv run python run_trial.py --screenshot setup.png         # the starting layout

Each prints the score as JSON, for example `{"score": 1.0, "red_blocks_in_bin": ["red_1", "red_2", "red_3"],
"scraps_lost": [], "arms_touched": false, "steps": 801}`.

**Runtime:** about 1 to 4 minutes per trial on an RTX 2070 Super, most of it rendering the two cameras every step.

**Seeds:** none to choose. The layout is fixed, and the second arm's routine is planned from it the same way every time,
so it moves the same way in every trial.

## Interface

`task_env.TaskEnv`: `reset()` and `step(action)` return the observation; `done` says when the trial is over; `score()`
gives the result.

- **Control:** 20 Hz, at most 1,500 steps (75 s). Inside each step the simulation steps at 50 Hz, the second arm with
  it; the arm under test holds its command for the step.
- **Action:** 7 numbers: six joint angle targets in radians, then the gripper, from 0 (closed) to 1 (open).
- **Observation:** `joints`, 13 floats (six joint angles, the gripper opening from 0 to 1, six joint speeds); `overhead`
  and `wrist`, 240 by 320 RGB images from a camera above the table and one on the controlled arm's wrist.
- **The controlled arm:** the left arm in the scene (positive y), which is the arm on the right of the screenshot, as it
  looks toward the arms.

## Scoring

`TaskEnv.score()`:

- The share of the three red blocks in the bin at the end.
- 0 if a white scrap is anywhere but the clutter and the grey patch (taken from the second arm), or if the arms touched
  at any point.

The trial stops at the first touch between the arms, and once the score has held at 1 for 1 s.

## The second arm

`Worker` in `sim/partner.py`. It sweeps the five lanes onto the grey patch on its side, nearest first, one every 10 s
from 12 s in, whatever is in them. Every move is planned when the trial starts, from the scene's fixed layout, and
played on the same clock in every trial: it reads nothing from the scene while it runs, does not wait for the tested arm
and does not move out of its way. Holding still scores 0.0: the second arm sweeps the red blocks onto its patch with the
scraps.

## The demo

`--policy demo` drives the controlled arm with `Helper` in `sim/partner.py` at 20 Hz: a `Picker` that takes the red
blocks still in the clutter, the one whose lane is swept soonest first; it waits while the worker's hand is near, or
while by its timetable the worker will reach across a lane near the block before the block could be lifted out. Unlike a
policy, it reads the simulator's true state (where everything on the table and the other arm are), not the cameras: it
shows that the task can be done with this arm in this scene. In the demo all three red blocks are in the bin by 29 s,
each before its lane is swept. With both arms scripted all three were rescued and no scrap lost, with no touch, in each
of 8 runs.

## Running Dream-1

`dream_run.py` runs Dream-1 (`dreamlabs/dream-1-beta`) on the task the way Dream Labs' own YAM MuJoCo example drives a
simulator: both arms' joints and gripper openings, and nine-frame windows from the YAM station's three calibrated
cameras (top, left wrist, right wrist), in the station's request keys. Dream-1 controls the left arm; its right-arm rows
are not executed, since that arm is the task's scripted second arm. Simulated time stands still while a request is out.

    uv sync --extra dream
    MUJOCO_GL=egl uv run python dream_run.py --key-file <file holding the API key> --out dream_run
    MUJOCO_GL=egl uv run python dream_run.py --dry-run                     # checks the request; no API call

Our run scored 0.0: Dream-1 never took a red block, and its arm touched the other arm at 33.0 s (28 requests of 24 rows
each).

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
  table (`scene.py`), inverse kinematics (`ik.py`), and the arms' scripts (`partner.py`).
- `assets/`: the YAM Pro arm and linear_4310 gripper model, from i2rt (MIT License; see above).