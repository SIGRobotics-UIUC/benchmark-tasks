# Sweep a table clear while another robot arm picks blocks off it

The arm under test picks up a hand broom and sweeps six pieces of clutter off the table. At the same time, a second
YAM arm takes four red blocks off the same table into its bin. The second arm doesn't wait for anyone or steer around
anything, so the sweeper has to fit its strokes into the gaps. The run is 30 seconds, about as long as the other arm's
job, so there's no free table to sweep at the end. Red blocks can be swept off too, and then they count like any other
item. Touching the other arm or the block in its hand scores 0.

## Running a trial

Python 3.11 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is used when present; on a Mac it runs on the CPU.

    uv sync
    MUJOCO_GL=egl uv run python run_trial.py --policy hold                  # the arm stays still: 0.0
    MUJOCO_GL=egl uv run python run_trial.py --policy demo --video demo.mp4  # the demonstration: 1.0
    MUJOCO_GL=egl uv run python run_trial.py --policy mymodule:MyPolicy      # your policy
    MUJOCO_GL=egl uv run python run_trial.py --screenshot setup.png          # the starting layout

(Leave out `MUJOCO_GL=egl` on a Mac.) Each one prints the score as JSON, for example
`{"score": 1.0, "items_gone": 10, "items_counted": 10, "clutter_left": 0, "red_binned": 3, "red_swept_off": 1, "contact": null, "steps": 569}`.

**Runtime:** about 1 to 2 minutes per trial on an M-series Mac's CPU, mostly spent rendering the two cameras.

**Seeds:** none. The layout is fixed, and the second arm moves the same way in every trial (`PARTNER_SEED = 0` in
`task_env.py`).

## Interface

`task_env.TaskEnv`: `reset()` and `step(action)` return the observation, `done` says when the trial is over, and
`score()` gives the result.

- **Control:** 20 Hz, 600 steps at most (30 s). The simulation runs at 50 Hz with the second arm in step; your arm
  holds each command for the whole control step.
- **Action:** 7 numbers: six joint angle targets in radians, then the gripper, from 0 (closed) to 1 (open).
- **Observation:** `joints`, 13 floats (six joint angles, the gripper opening from 0 to 1, six joint speeds), and
  `overhead` and `wrist`, 240 x 320 RGB images from a camera above the table and one on your arm's wrist.
- **Your arm:** the left arm in the scene (positive y). In the screenshot, which looks toward the arms, it's on the
  right.

## Scoring

`TaskEnv.score()`:

- Items off the table divided by items counted. The 6 clutter items always count. A red block counts only if it ends
  up off the table. Red blocks the other arm bins, or that stay on the table, are left out, so leaving them alone costs
  nothing and the other arm's work earns you nothing.
- 0 if your arm or the broom touches the other arm, or touches the block the other arm is holding. The trial stops at
  the first touch.
- The trial ends at 600 steps, or earlier once everything counted is gone and the other arm has finished.

## The demo

`--policy demo` drives your arm with the scripted `Sweeper` in `sim/partner.py`. It isn't a policy: it reads the
simulator's true state, including where the other arm's script is about to move. All it shows is that the task can be
done with this arm in this scene. It picks the broom up by its grip and makes one stroke toward its own side edge
through each 20 cm band of the table. Before each stroke it checks that neither arm will come within 9 cm of the
other, and if they would, it holds still until the other arm has passed. When the table is clear it stands the broom
back where it started. It scored 1.0 and finished in 569 steps. We also ran it 3 more times with every item and
block moved up to 1 cm and the other arm starting up to 0.3 s early or late; it scored 1.0 each time.

## Running Dream-1

`dream_run.py` runs Dream-1 (`dreamlabs/dream-1-beta`) the way Dream Labs' own YAM MuJoCo example does. Dream-1
controls your arm; its rows for the other arm aren't executed. The prompt is the instruction from `task.json`.

    uv sync --extra dream
    MUJOCO_GL=egl uv run python dream_run.py --out dream_run       # needs DREAM_API_KEY; uses credits
    MUJOCO_GL=egl uv run python dream_run.py --dry-run             # checks the request; no API call

## Files

- `task_env.py`: the task, the second arm and the scoring.
- `run_trial.py`: runs and scores one trial; also makes the demo, the screenshot and videos.
- `dream_run.py`: runs Dream-1 on the task.
- `cameras.py`: the policy's two cameras, the YAM station's three cameras for Dream-1, and the wide view used for the
  screenshot and videos (the policy doesn't get this one).
- `sim/`: the scene and objects (`benchmark.py`, `props.py`), the YAM model and its controller (`robot.py`,
  `env_cfg.py`), the table (`scene.py`), inverse kinematics (`ik.py`), and the scripts for the second arm and the demo
  (`partner.py`).
- `assets/`: the YAM Pro arm and linear_4310 gripper model, from i2rt.
- `media/demo.mp4`: the demo run (`--policy demo`), scoring 1.0.
