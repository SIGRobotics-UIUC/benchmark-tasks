"""Run and score one trial.

    python run_trial.py --policy hold                    # the arm stays still: scores 0
    python run_trial.py --policy demo --video demo.mp4   # the full-score demonstration
    python run_trial.py --policy mymodule:MyPolicy       # yours: MyPolicy().act(obs) -> 7 numbers
    python run_trial.py --screenshot setup.png           # the starting layout, from a wide view of the table

Prints the score as JSON.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib

import mediapy
import numpy as np

from cameras import Overview
from task_env import CONTROL_HZ, READY, TaskEnv

HERE = pathlib.Path(__file__).resolve().parent


class Hold:
  def act(self, obs):
    return np.r_[READY, 1.0]


class Demo:
  """The recorded full-score run of the controlled arm, replayed at the task's control rate."""

  def __init__(self) -> None:
    replay = np.load(HERE / "data" / "recorded_rows.npz")
    self.rows, self.dt, self.t = replay["left"], float(replay["dt"]), 0

  def act(self, obs):
    self.t += 1
    return self.rows[min(int(round(self.t / CONTROL_HZ / self.dt)), len(self.rows) - 1)]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy", default="hold")
  parser.add_argument("--video")
  parser.add_argument("--screenshot")
  args = parser.parse_args()
  env = TaskEnv()
  obs = env.reset()
  view = Overview(env.env) if args.video or args.screenshot else None
  if args.screenshot:
    mediapy.write_image(args.screenshot, view.render())
  if args.policy == "hold":
    policy = Hold()
  elif args.policy == "demo":
    policy = Demo()
  else:
    module, name = args.policy.split(":")
    policy = getattr(importlib.import_module(module), name)()
  frames = [view.render()] if args.video else []
  while not env.done:
    obs = env.step(policy.act(obs))
    if args.video and env.steps % 2 == 0:
      frames.append(view.render())
  if args.video:
    mediapy.write_video(args.video, frames, fps=CONTROL_HZ / 2, codec="h264")
  print(json.dumps(env.score()))


if __name__ == "__main__":
  main()
