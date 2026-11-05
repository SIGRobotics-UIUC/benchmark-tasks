# Pick blocks off a table while another robot arm sweeps it clear

Your arm puts four red blocks in its bin. At the same time, a second YAM arm sweeps the table clear with a hand broom.
The second arm won't wait or steer around you, and its strokes run through two of the blocks, so you have to work
between its strokes and get out of its way when it comes. The trial lasts 30 seconds, so waiting for it to finish
doesn't leave enough time. Touching the other arm or its broom scores 0.

## Running a trial

Needs Python 3.11 and [uv](https://docs.astral.sh/uv/). Uses a CUDA GPU if there is one, otherwise the CPU.

    uv sync
    MUJOCO_GL=egl uv run python run_trial.py --policy hold                  # arm stays still: 0.0
    MUJOCO_GL=egl uv run python run_trial.py --policy demo --video demo.mp4  # the demo: 1.0
    MUJOCO_GL=egl uv run python run_trial.py --policy mymodule:MyPolicy      # your policy
    MUJOCO_GL=egl uv run python run_trial.py --screenshot setup.png          # starting layout

On a Mac, leave out `MUJOCO_GL=egl`. Each command prints the score as JSON, for example:

    {"score": 1.0, "red_binned": 4, "red_swept_off": 0, "clutter_left": 0, "contact": null, "steps": 545}

**Runtime:** about 1 to 2 minutes per trial on an M-series Mac CPU, mostly rendering the cameras.

**Seeds:** none. The layout is fixed, and the second arm moves the same way every trial unless the clutter is moved.

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
the table, going straight from one to the next, sweeps again at any clutter left, and stands the broom back up, by
about 20 s. It never waits for or steers around your arm. Holding still scores 0.0: two blocks get swept off, and
nothing goes in the bin. Taking one block and then waiting for the second arm to finish scored 0.75.

## The demo

`--policy demo` drives your arm with the scripted `Picker` in `sim/partner.py`. It isn't a policy: it reads the
simulator's true state, including where the other arm is about to move. It only shows the task can be done.

It takes the two blocks in the broom's path first, then the two near the middle. Whenever the other arm or its broom
would come within 12 cm (between the arms' centre lines) in the next 0.6 s, it backs off over its bin, dropping any
block it holds, and tries that block again last. It scored 1.0 in 545 steps.

## Running Dream-1

Not run. `dream_run.py` runs Dream-1 (`dreamlabs/dream-1-beta`) the way Dream Labs' YAM MuJoCo example does. Dream-1
controls your arm, its rows for the other arm are ignored, and the prompt is the instruction in `task.json`.

    uv sync --extra dream
    MUJOCO_GL=egl uv run python dream_run.py --out dream_run       # needs DREAM_API_KEY; uses credits
    MUJOCO_GL=egl uv run python dream_run.py --dry-run             # checks the request; no API call

## Files

- `task_env.py`: the task, the second arm and the scoring.
- `run_trial.py`: runs and scores one trial; also the demo, screenshot and videos.
- `dream_run.py`: runs Dream-1 on the task.
- `cameras.py`: the policy's two cameras, the YAM station's three cameras for Dream-1, and the wide view for the
  screenshot and videos (not given to the policy).
- `sim/`: the scene and objects (`benchmark.py`, `props.py`), the YAM model and controller (`robot.py`,
  `env_cfg.py`), the table (`scene.py`), inverse kinematics (`ik.py`), and the scripts for the second arm and the demo
  (`partner.py`).
- `assets/`: the YAM Pro arm and linear_4310 gripper model, from i2rt.
- `media/demo.mp4`: the demo run, scoring 1.0.
