# Unload fruit from a bin another robot arm is also unloading

One YAM arm, the one under test, takes the fruit out of a shared bin into its tray while a second YAM arm takes the
utensils out of the same bin. The second arm does its own job and does not react: it replays a recorded run, so every
trial sees the same moves at the same times. The first touch between the arms ends the trial at 0.

## Running a trial

Python 3.11 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is used when present.

    uv sync
    MUJOCO_GL=egl uv run python run_trial.py --policy hold                 # the arm stays still: 0.0
    MUJOCO_GL=egl uv run python run_trial.py --policy demo --video demo.mp4 # the full-score demo: 1.0
    MUJOCO_GL=egl uv run python run_trial.py --policy mymodule:MyPolicy     # your policy
    MUJOCO_GL=egl uv run python run_trial.py --screenshot setup.png         # the starting layout

Each prints the score as JSON, for example
`{"score": 1.0, "fruit_in_tray": 3, "arms_touched": false, "utensil_off_table": false, "steps": 1800}`.

**Runtime:** about 2 minutes per trial for the full 1,800 steps on an RTX 2070 Super, video included. A trial that
ends at a touch takes seconds.

**Seeds:** none to choose. The layout is fixed (`LAYOUT_SEED = 0` in `task_env.py`), and the second arm's motion is a
fixed recording (`data/recorded_rows.npz`).

## Interface

`task_env.TaskEnv`: `reset()` and `step(action)` return the observation; `done` says when the trial is over; `score()`
gives the result.

- **Control:** 20 Hz, at most 1,800 steps.
- **Action:** 7 numbers: six joint angle targets in radians, then the gripper, from 0 (closed) to 1 (open).
- **Observation:** `joints`, 13 floats (six joint angles, the gripper opening from 0 to 1, six joint speeds); `overhead`
  and `wrist`, 240 by 320 RGB images from a camera above the table and one on the controlled arm's wrist.
- **The controlled arm:** the left arm in the scene (positive y), which is the arm on the right of the screenshot, as it
  looks toward the arms.

## Scoring

`TaskEnv.score()`, read at the end of the trial:

- 1/3 for each of the apple, orange and banana whose centre is inside the green tray's inner walls (20 x 16 cm), resting
  (less than 8 cm above the table).
- 0 if the arms touched at any point. The trial stops at the first contact between any parts of the two arms.
- 0 if any utensil ended up more than 5 cm below the table top, that is, off the table: wrecking the other arm's job
  never scores.

## The demo

`--policy demo` replays the controlled arm's joint targets from a recorded run, at 20 Hz, in which a camera-only policy
put all three fruit in the tray beside the second arm with no contact. It scores 1.0 every time we have run it.

## Running Dream-1

`dream_run.py` runs Dream-1 (`dreamlabs/dream-1-beta`) on the task the way Dream Labs' own YAM MuJoCo example drives a
simulator: both arms' joints and gripper openings, and nine-frame windows from the YAM station's three calibrated
cameras (top, left wrist, right wrist), in the station's request keys. Dream-1 controls the left arm; its right-arm
rows are not executed, since that arm replays its recording. Simulated time stands still while a request is out.

    uv sync --extra dream
    MUJOCO_GL=egl uv run python dream_run.py --key-file <file holding the API key> --out dream_run
    MUJOCO_GL=egl uv run python dream_run.py --dry-run                     # checks the request; no API call

Our run scored 0.0: the arm touched the other arm at step 149 (7.5 s), after 7 requests of 24 rows each.

The station cameras are mounted from the SDK's calibration (`dream_examples/yam_mujoco/assets/sim-eval-yam.json`):
the wrist cameras on `link6`, turned by the fixed rotation between that SDK model's `link_6` frame and this one's, and
the top camera at its place relative to the midpoint of the two arm bases. This arm model's kinematics agree with the
SDK's YAM to within 1.4 cm at its 22 reference poses. The arm bases here are 50 cm apart, against the station's 61 cm.

## Files

- `task_env.py`: the task, the second arm's replay, and the scoring.
- `run_trial.py`: runs and scores one trial; also the demo, the screenshot and videos.
- `dream_run.py`: runs Dream-1 on the task (needs `uv sync --extra dream` and an API key).
- `cameras.py`: the policy's two cameras, the YAM station's three cameras for Dream-1, and the wide view used for the
  screenshot and videos (not given to the policy).
- `sim/`: the scene: the YAM model with its controller, the table, bin, trays and objects.
- `assets/`: the YAM Pro arm and linear_4310 gripper model, from i2rt.
- `data/recorded_rows.npz`: the recorded run: the second arm's joint targets and the demo's, every 20 ms.
