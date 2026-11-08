# YAM Pro + linear_4310 gripper

The rig's real arm and gripper, as i2rt models them. Generated, not hand-written:

    i2rt.robots.utils.combine_arm_and_gripper_xml(ArmType.YAM_PRO, GripperType.LINEAR_4310)

with two edits: mesh paths made relative to `meshes/` (copied from
`i2rt/robot_models`), and the arm wrapped in one root body, `arm`, so the base mesh
moves with the rest when the entity is placed.

Collision pads, the fitted link shells, actuators and gravity compensation are
added in code (`sim/robot.py`); every mesh here is visual only.

## Source and license

The model and its meshes are I2RT Robotics' YAM Pro arm and linear_4310 gripper, from i2rt
(https://github.com/i2rt-robotics/i2rt, `i2rt/robot_models`), copyright (c) I2RT Robotics, used under the MIT License.
The license text, with the copyright notice, is in `LICENSE` in this folder. Our changes are the two edits above.
