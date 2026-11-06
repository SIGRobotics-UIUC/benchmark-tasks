# Drive your own screw with a screwdriver another robot arm also needs

One YAM arm, the one under test, and a second YAM arm each have a wooden block in front of them with a screw in its
top, and there is one screwdriver standing in a holder between them. The tested arm must take the screwdriver, drive
its own screw half a turn with it, leave the other screw alone, and stand the screwdriver back in its holder, because
the second arm needs it too. A screw turns only with the cross point seated in its slot and the driver turned. The
second arm waits for the screwdriver to stand in its holder with the tested arm's hand away from it, takes it, drives
its own screw and stands it back. The arms must never touch.

## Running a trial

Python 3.11 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is used when present.

    uv sync
    MUJOCO_GL=egl uv run python run_trial.py --policy hold                 # the arm stays still: 0.0
    MUJOCO_GL=egl uv run python run_trial.py --policy demo --video demo.mp4 # the demonstration: 1.0
    MUJOCO_GL=egl uv run python run_trial.py --policy mymodule:MyPolicy     # your policy
    MUJOCO_GL=egl uv run python run_trial.py --screenshot setup.png         # the starting layout

Each prints the score as JSON, for example
`{"score": 1.0, "screw_turned_deg": 212.1, "driver_back": true, "turned_other_screw": false, "arms_touched": false, "steps": 908}`.

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
  looks toward the arms. Its screw is the one in front of it.

## Scoring

`TaskEnv.score()`:

- How far the controlled arm's screw has been turned clockwise, seen from above, over half a turn, up to 1. The
  screw's head turns on a hinge against a fixed friction of 0.05 N m; the cross point turns it by pushing on the four
  quarters of the head around the slot.
- Halved if the screwdriver is not back in its holder at the end (standing in it, held by nobody): the other arm
  needs it.
- 0 if the other screw turned while the controlled arm held the screwdriver and the other arm did not, or if the arms
  touched at any point. The trial stops at the first touch or wrong turn, and once the screw is half a turn in and the
  screwdriver has been back for 1 s.

## The demo

`--policy demo` drives the controlled arm with the same script as the second arm (`Driver` in `sim/partner.py`), at
20 Hz. Unlike a policy, it reads the simulator's true state (where the screwdriver, the screws and the other arm are),
not the cameras: it shows that the task can be done with this arm in this scene. It waits a random while, takes the
screwdriver by the handle with the fingers straight down only when it stands in its holder and the other hand is not at
it, and carries it without turning the wrist to just over its screw. There it turns the wrist to line the cross point up
with the slot, lowers it in and twists a quarter turn, lifts, and again until the screw is half a turn in (three
twists: the point slips a little in each), then stands the screwdriver back. With both arms scripted both screws went in
and the screwdriver went back, without the arms touching, in each of 8 runs with different random waits.

## Running Dream-1

`dream_run.py` runs Dream-1 (`dreamlabs/dream-1-beta`) on the task the way Dream Labs' own YAM MuJoCo example drives a
simulator: both arms' joints and gripper openings, and nine-frame windows from the YAM station's three calibrated
cameras (top, left wrist, right wrist), in the station's request keys. Dream-1 controls the left arm; its right-arm
rows are not executed, since that arm is the task's scripted second arm. Simulated time stands still while a request is
out.

    uv sync --extra dream
    MUJOCO_GL=egl uv run python dream_run.py --key-file <file holding the API key> --out dream_run
    MUJOCO_GL=egl uv run python dream_run.py --dry-run                     # checks the request; no API call

Our run scored 0.0: in 3 minutes (150 requests of 24 rows each) Dream-1 never took the screwdriver; its arm reached up and back over its own base and stayed there, while the other arm took the screwdriver, drove its own screw and stood the screwdriver back. The arms never touched.

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
