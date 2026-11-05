# Sweep a table clear while another robot arm picks blocks off it

Your arm picks up a hand broom and sweeps six pieces of clutter off the table. At the same time, a second YAM arm
moves four red blocks from the same table into its bin. The second arm won't wait or steer around you, so you have to
sweep in the gaps between its reaches. The trial lasts 30 seconds, about as long as the other arm's job, so waiting
for it to finish doesn't help. Touching the other arm or the block it's holding scores 0.

## Running a trial

Needs Python 3.11 and [uv](https://docs.astral.sh/uv/). Uses a CUDA GPU if there is one, otherwise the CPU.

    uv sync
    MUJOCO_GL=egl uv run python run_trial.py --policy hold                  # arm stays still: 0.0
    MUJOCO_GL=egl uv run python run_trial.py --policy demo --video demo.mp4  # the demo: 1.0
    MUJOCO_GL=egl uv run python run_trial.py --policy mymodule:MyPolicy      # your policy
    MUJOCO_GL=egl uv run python run_trial.py --screenshot setup.png          # starting layout

On a Mac, leave out `MUJOCO_GL=egl`. Each command prints the score as JSON, for example:

    {"score": 1.0, "items_gone": 6, "items_counted": 6, "clutter_left": 0, "red_binned": 4, "red_swept_off": 0, "contact": null, "steps": 569}

**Runtime:** about 1 to 2 minutes per trial on an M-series Mac CPU, mostly rendering the cameras.

**Seeds:** none. The layout is fixed, and the second arm moves the same way every trial.

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

- Score = items swept off the table / items counted.
- All 6 clutter items count. A red block counts only if it ends up off the table; red blocks in the bin or still on
  the table are left out. The other arm's work never earns you anything, and leaving red blocks alone never costs you.
- 0 if your arm or the broom touches the other arm or the block it's holding. The trial stops at the first touch.
- The trial ends at 600 steps, or earlier once every counted item is gone and the other arm has finished.

## The demo

`--policy demo` drives your arm with the scripted `Sweeper` in `sim/partner.py`. It isn't a policy: it reads the
simulator's true state, including where the other arm is about to move. It only shows the task can be done.

It picks up the broom and makes one stroke toward its own edge through each 20 cm band of the table. Before each
stroke, it checks that the two arms will stay at least 9 cm apart, and waits if they won't. Then it stands the broom
back up. It scored 1.0 in 569 steps. In 3 more runs, with every item and block moved up to 1 cm and the other arm
starting up to 0.3 s early or late, it scored 1.0 each time.

## Dream-1

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
