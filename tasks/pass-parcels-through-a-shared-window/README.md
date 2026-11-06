# Pass parcels through a window another robot arm is also passing through

A wall 22 cm high runs down the middle of the table between the two YAM arms, with one opening in it, 16 cm wide and
open to the top, like a serving hatch. The arm under test has two red cubes on a mat on its side; they go into a tray
just through the opening on the other side. The second arm passes its two blue cylinders the other way, into a tray on
the tested arm's side. The opening takes one hand at a time: the second arm keeps its hand out of it while the tested
arm's hand is there, and takes its parcel back to wait if the tested arm's hand comes to the opening while it is on its
way. A parcel must go through the opening, low: one lifted higher than the top of the wall does not count. The arms
must never touch.

## Running a trial

Python 3.11 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is used when present.

    uv sync
    MUJOCO_GL=egl uv run python run_trial.py --policy hold                 # the arm stays still: 0.0
    MUJOCO_GL=egl uv run python run_trial.py --policy demo --video demo.mp4 # the demonstration: 1.0
    MUJOCO_GL=egl uv run python run_trial.py --policy mymodule:MyPolicy     # your policy
    MUJOCO_GL=egl uv run python run_trial.py --screenshot setup.png         # the starting layout

Each prints the score as JSON, for example
`{"score": 1.0, "passed": ["left_parcel_1", "left_parcel_2"], "lifted_over_wall": [], "arms_touched": false, "other_parcels_passed": 2, "steps": 553}`.

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
  looks toward the arms. Its parcels are the red cubes on the green mat beside it, and its tray the green one beyond the
  wall.

## Scoring

`TaskEnv.score()`:

- The share of the two red cubes in the green tray at the end (inside its walls, below 6 cm over the table), not
  counting a cube that was higher than the wall's top at any moment: it went over the wall, not through the window.
- 0 if the arms touched at any point. The trial stops at the first touch, and once both cubes have been in the tray
  for 1 s.

The JSON lists the cubes that went over the wall as `lifted_over_wall`.

## The demo

`--policy demo` drives the controlled arm with the same script as the second arm (`Passer` in `sim/partner.py`), at
20 Hz. Unlike a policy, it reads the simulator's true state (where the parcels, the wall and the other arm are), not
the cameras: it shows that the task can be done with this arm in this scene. It waits a random while; then, parcel by
parcel, it picks the parcel up a little way, waits while the other hand is in the opening, carries it low to just in
front of the opening, through it to just over its tray, lets go and comes back out, then goes up and by way of ready to
the next. With both arms scripted all four parcels went through, without the arms touching or a parcel going over the
wall, in each of 8 runs with different random waits.

## Running Dream-1

`dream_run.py` runs Dream-1 (`dreamlabs/dream-1-beta`) on the task the way Dream Labs' own YAM MuJoCo example drives a
simulator: both arms' joints and gripper openings, and nine-frame windows from the YAM station's three calibrated
cameras (top, left wrist, right wrist), in the station's request keys. Dream-1 controls the left arm; its right-arm
rows are not executed, since that arm is the task's scripted second arm. Simulated time stands still while a request is
out.

    uv sync --extra dream
    MUJOCO_GL=egl uv run python dream_run.py --key-file <file holding the API key> --out dream_run
    MUJOCO_GL=egl uv run python dream_run.py --dry-run                     # checks the request; no API call

Our run scored 0.5: in 3 minutes (150 requests of 24 rows each) Dream-1 spent most of the run over its mat, nudging its cubes, and near the end picked one up and carried it through the opening, below the wall's top, into the green tray; the other stayed on the mat. The other arm passed both its cylinders, and the arms never touched.

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
