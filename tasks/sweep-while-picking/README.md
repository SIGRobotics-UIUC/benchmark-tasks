# Sweep a work area clear while another robot arm picks blocks from it

Your arm picks up a hand broom and sweeps six pieces of clutter past a white line on its side of the table. At the
same time, a second YAM arm moves four red blocks from the same table into its bin. The second arm won't wait or steer
around you, so you have to sweep in the gaps between its reaches. The trial lasts 30 seconds, about as long as the
other arm's job, so waiting for it to finish doesn't help. Touching the other arm or the block it's holding scores 0.

## Running a trial

Needs Python 3.11 and [uv](https://docs.astral.sh/uv/). Uses a CUDA GPU if there is one, otherwise the CPU.

    uv sync
    MUJOCO_GL=egl uv run python run_trial.py --policy hold                  # arm stays still: 0.0
    MUJOCO_GL=egl uv run python run_trial.py --policy demo --video demo.mp4  # the demo: 1.0
    MUJOCO_GL=egl uv run python run_trial.py --policy mymodule:MyPolicy      # your policy
    MUJOCO_GL=egl uv run python run_trial.py --screenshot setup.png          # starting layout

On a Mac, leave out `MUJOCO_GL=egl`. Each command prints the score as JSON, for example:

    {"score": 1.0, "items_gone": 6, "items_counted": 6, "clutter_left": 0, "red_binned": 4, "red_swept_off": 0, "contact": null, "steps": 568}

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

- Score = clutter items past the white line / 6. An item is past it when its centre is more than 45 cm from the
  middle toward your arm and it is still on the table. The table runs on 30 cm past the line, with a 3 cm lip
  around that part's open edges so nothing goes over them.
- Red blocks don't count either way: binned by the other arm, left on the table or swept off, they neither earn nor
  cost you anything.
- 0 if your arm or the broom touches the other arm or the block it's holding. The trial stops at the first touch.
- The trial ends at 600 steps, or earlier once all the clutter is past the line and the other arm has finished.

In real trials, an operator stands by to stop both arms at the first touch.

## The second arm

`Picker` in `sim/partner.py`. It waits 2 s, then reaches for each red block's starting spot in turn and drops it in
its bin, one every 6.2 s, the last at about 27 s. Every move is planned when the trial starts, from the scene's fixed
layout, and played on the same clock in every trial: it reads nothing from the scene while it runs, reaches for a block
whether or not it is still there, and never waits for or steers around your arm. Holding still scores 0.0.

## The demo

`--policy demo` drives your arm with the scripted `Sweeper` in `sim/partner.py`. It isn't a policy: it reads the
simulator's true state, including where the other arm is about to move. It only shows the task can be done.

It picks up the broom and makes one stroke toward the line through each 20 cm band of the table. Before each
stroke, it checks that the two arms will stay at least 9 cm apart, and waits if they won't. It sweeps again at any
clutter left, then stands the broom back up. It scored 1.0 in 568 steps. In 8 more runs, with every item and block
moved up to 1 cm and the other arm starting up to 0.3 s early or late, it scored 1.0 in 7; in the eighth the
simulation became unstable (a ball pinched at 21.5 s) and the trial was stopped as invalid.

## Running Dream-1

`dream_run.py` runs Dream-1 (`dreamlabs/dream-1-beta`) the way Dream Labs' YAM MuJoCo example does. Dream-1 controls
your arm, its rows for the other arm are ignored, and the prompt is the instruction in `task.json`. It saves the run's
video, its score and every action chunk Dream-1 returned (`dream_run/`), retries a dropped connection up to 3 times,
and carries on if Dream Labs replaces an idle session.

    uv sync --extra dream
    MUJOCO_GL=egl uv run python dream_run.py --key-file <file holding the key> --out dream_run   # uses credits
    MUJOCO_GL=egl uv run python dream_run.py --dry-run                                         # no API call

Our run scored 0.0: Dream-1's arm touched the other arm 4.6 s in (step 91, 4 requests of 24 rows), before any
clutter moved. An earlier version of the instruction, "sweep the clutter off the table", was refused by Dream Labs'
API as unsafe (sweeping things onto the floor), which is why the swept clutter now stays on the table, past a line.

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
