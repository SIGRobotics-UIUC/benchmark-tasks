# Pick blocks off a table while another robot arm sweeps it clear

Your arm puts four red blocks in its bin. At the same time, a second YAM arm sweeps the table clear with a hand broom.
The second arm works to a fixed routine: it won't wait or steer around you, and its strokes come down right on two
of the blocks, so you have to get them clear of the broom in time and keep out of its way when it comes. The trial
lasts 30 seconds, so waiting for it to finish doesn't leave enough time. Touching the other arm or its broom scores 0.

## Running a trial

Needs Python 3.11 and [uv](https://docs.astral.sh/uv/). Uses a CUDA GPU if there is one, otherwise the CPU.

    uv sync
    MUJOCO_GL=egl uv run python run_trial.py --policy hold                  # arm stays still: 0.0
    MUJOCO_GL=egl uv run python run_trial.py --policy demo --video demo.mp4  # the demo: 1.0
    MUJOCO_GL=egl uv run python run_trial.py --policy mymodule:MyPolicy      # your policy
    MUJOCO_GL=egl uv run python run_trial.py --screenshot setup.png          # starting layout

On a Mac, leave out `MUJOCO_GL=egl`. Each command prints the score as JSON, for example:

    {"score": 1.0, "red_binned": 4, "red_swept_off": 0, "clutter_left": 0, "contact": null, "steps": 575}

**Runtime:** about 1 to 2 minutes per trial on an M-series Mac CPU, mostly rendering the cameras.

**Seeds:** none. The layout is fixed, and the second arm's routine is planned from it the same way every time, so it
moves the same way in every trial.

## Interface

`task_env.TaskEnv`: `reset()` and `step(action)` return the observation, `done` says when the trial is over, and
`score()` returns the result.

- **Control:** 20 Hz, at most 600 steps (30 s). The simulation runs at 50 Hz; your arm holds each command for the
  whole control step.
- **Action:** 7 numbers: six joint angle targets in radians, then the gripper, from 0 (closed) to 1 (open).
- **Observation:** `joints` (13 floats: six joint angles, gripper opening from 0 to 1, six joint speeds), and
  `overhead` and `wrist`, 240 x 320 RGB images from a camera above the table and one on your wrist.
- **Your arm:** the left arm in the scene (positive y). It's on the right in the screenshot.

## Scoring

- Score = red blocks resting in the bin / 4. Blocks swept off the table count as not binned.
- 0 if your arm, or a block in your gripper (within 3 cm of the grasp point), touches the other arm or its broom. The
  trial stops at the first touch.
- The trial ends at 600 steps, or earlier once every red block is in the bin or off the table and the other arm has
  finished.

## The second arm

`Sweeper` in `sim/partner.py`. It picks up the broom, makes one stroke toward its own edge through each 20 cm band of
the table, going straight from one to the next, and stands the broom back up, by about 19 s. Every move is planned
when the trial starts, from the scene's fixed layout, and played on the same clock in every trial: it reads nothing
from the scene while it runs, and never waits for or steers around your arm. Holding still scores 0.0: nothing goes in
the bin, and the broom comes down on the two blocks in its path and shoves them several centimetres.

## The demo

`--policy demo` drives your arm with the scripted `Picker` in `sim/partner.py`. It isn't a policy: it reads the
simulator's true state, including where the other arm is about to move. It only shows the task can be done.

First, before the broom's first stroke, it nudges the two blocks in the broom's path across to its own side with its
closed fingers. Then it bins all four, starting each pick only when, by the other arm's fixed timetable, the whole
pick and carry stays at least 12 cm (between the arms' centre lines) from the other arm and its broom; while it
waits between picks, it backs off over its bin if the other arm comes that close. It scored 1.0 in 575 steps. In 8
more runs, with every item and block moved up to 1 cm and the other arm starting up to 0.3 s early or late, it scored
1.0 each time.

## Running Dream-1

`dream_run.py` runs Dream-1 (`dreamlabs/dream-1-beta`) the way Dream Labs' YAM MuJoCo example does. Dream-1 controls
your arm, its rows for the other arm are ignored, and the prompt is the instruction in `task.json`. It saves the run's
video, its score and every action chunk Dream-1 returned (`dream_run/`), retries a dropped connection up to 3 times,
and carries on if Dream Labs replaces an idle session.

    uv sync --extra dream
    MUJOCO_GL=egl uv run python dream_run.py --key-file <file holding the key> --out dream_run   # uses credits
    MUJOCO_GL=egl uv run python dream_run.py --dry-run                                         # no API call

Our run scored 0.0: in the full 30 s (25 requests of 24 rows) Dream-1 put no block in the bin, one block was swept
off, and the arms never touched.

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

- `task_env.py`: the task, the second arm and the scoring.
- `run_trial.py`: runs and scores one trial; also the demo, screenshot and videos.
- `dream_run.py`: runs Dream-1 on the task.
- `cameras.py`: the policy's two cameras, the YAM station's three cameras for Dream-1, and the wide view for the
  screenshot and videos (not given to the policy).
- `sim/`: the scene and objects (`benchmark.py`, `props.py`), the YAM model and controller (`robot.py`,
  `env_cfg.py`), the table (`scene.py`), inverse kinematics (`ik.py`), and the scripts for the second arm and the demo
  (`partner.py`).
- `assets/`: the YAM Pro arm and linear_4310 gripper model, from i2rt (MIT License; see above).
- `media/demo.mp4`: the demo run, scoring 1.0.
