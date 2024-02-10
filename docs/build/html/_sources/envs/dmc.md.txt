# DeepMind Control (DMC)

<!-- TODO: Add Vid -->

These are the Environment Wrappers for selected
[DeepMind Control](https://github.com/google-deepmind/dm_control)
environments in order to use our Motion Primitive gym interface with them.

## Step-Based Environments

When installing fancy_gym with the optional dmc extra (e.g. `pip install fancy_gym[dmc]`), all regular dmc tasks are avaible via [Shimmy](https://github.com/Farama-Foundation/Shimmy).

| Name                                    | Description                                                                                                             | Action Dim | Observation Dim |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------- | --------------- |
| `dm_control/acrobot-swingup-v0`         | Underactuated double pendulum (Acrobot) with torque applied to the second joint to swing up and balance.                | 1          | 6               |
| `dm_control/acrobot-swingup_sparse-v0`  | Similar to `acrobot-swingup-v0`, but with sparse rewards for achieving the swingup task.                                | 1          | 6               |
| `dm_control/ball_in_cup-catch-v0`       | Planar ball-in-cup task where a receptacle must swing to catch a ball. Sparse reward for catching.                      | 2          | 8               |
| `dm_control/cartpole-balance-v0`        | Cart-pole task where the goal is to balance an unactuated pole by moving a cart, starting near upright.                 | 1          | 5               |
| `dm_control/cartpole-balance_sparse-v0` | Similar to `cartpole-balance-v0`, but with sparse rewards.                                                              | 1          | 5               |
| `dm_control/cartpole-swingup-v0`        | Cart-pole task with the pole starting downward, requiring swinging up and balancing.                                    | 1          | 5               |
| `dm_control/cartpole-swingup_sparse-v0` | Similar to `cartpole-swingup-v0`, but with sparse rewards for the swingup task.                                         | 1          | 5               |
| `dm_control/cartpole-two_poles-v0`      | Extension of the Cart-pole domain with two serially connected poles, increasing the balancing challenge.                | 1          | 8               |
| `dm_control/cartpole-three_poles-v0`    | Extension of the Cart-pole domain with three serially connected poles, further increasing the challenge.                | 1          | 11              |
| `dm_control/cheetah-run-v0`             | Planar bipedal cheetah robot tasked with running. The reward is proportional to forward velocity up to a maximum speed. | 6          | 17              |
| `dm_control/dog-stand-v0`               | Dog robot task focusing on achieving a standing posture.                                                                | 38         | 223             |
| `dm_control/dog-walk-v0`                | Dog robot tasked with walking, requiring coordination of movements.                                                     | 38         | 223             |
| `dm_control/dog-trot-v0`                | Dog robot performing a trotting gait.                                                                                   | 38         | 223             |
| `dm_control/dog-run-v0`                 | Dog robot tasked with running, combining speed with stability.                                                          | 38         | 223             |
| `dm_control/dog-fetch-v0`               | Dog robot playing fetch, involving locomotion and object interaction.                                                   | 38         | 232             |
| `dm_control/finger-spin-v0`             | Finger robot required to rotate an unactuated body on a hinge.                                                          | 2          | 9               |
| `dm_control/finger-turn_easy-v0`        | Finger robot task to align the tip of a free body with a target, easier version with a larger target.                   | 2          | 12              |
| `dm_control/finger-turn_hard-v0`        | Similar to `finger-turn_easy-v0`, but with a smaller target for increased difficulty.                                   | 2          | 12              |
| `dm_control/fish-upright-v0`            | Fish robot task focused on righting itself in a fluid environment.                                                      | 5          | 21              |
| `dm_control/fish-swim-v0`               | Fish robot swimming to a target, incorporating fluid dynamics.                                                          | 5          | 24              |
| `dm_control/hopper-stand-v0`            | One-legged hopper robot tasked with achieving a minimal torso height.                                                   | 4          | 15              |
| `dm_control/hopper-hop-v0`              | One-legged hopper robot required to hop, combining height and forward velocity.                                         | 4          | 15              |
| `dm_control/humanoid-stand-v0`          | Simplified humanoid robot maintaining a standing posture.                                                               | 21         | 67              |
| `dm_control/humanoid-walk-v0`           | Simplified humanoid robot walking at a specified speed.                                                                 | 21         | 67              |
| `dm_control/humanoid-run-v0`            | Simplified humanoid robot running, aiming for a high horizontal speed.                                                  | 21         | 67              |
| `dm_control/humanoid-run_pure_state-v0` | Simplified humanoid robot running with a focus on pure state control.                                                   | 21         | 55              |
| `dm_control/humanoid_CMU-stand-v0`      | Advanced humanoid robot (CMU model) maintaining a standing posture.                                                     | 56         | 137             |
| `dm_control/humanoid_CMU-walk-v0`       | Advanced humanoid robot (CMU model) walking.                                                                            | 56         | 137             |
| `dm_control/humanoid_CMU-run-v0`        | Advanced humanoid robot (CMU model) running.                                                                            | 56         | 137             |
| `dm_control/lqr-lqr_2_1-v0`             | Linear quadratic regulator (LQR) task with 2 masses and 1 actuator, focusing on position and control optimization.      | 1          | 4               |
| `dm_control/lqr-lqr_6_2-v0`             | Linear quadratic regulator (LQR) task with 6 masses and 2 actuators, more complex control optimization challenge.       | 2          | 12              |
| `dm_control/manipulator-bring_ball-v0`  | Planar manipulator robot bringing a ball to a target location, with object initialization variations.                   | 5          | 44              |
| `dm_control/manipulator-bring_peg-v0`   | Planar manipulator task of bringing a peg to a target peg.                                                              | 5          | 44              |
| `dm_control/manipulator-insert_ball-v0` | Planar manipulator task requiring inserting a ball into a basket.                                                       | 5          | 44              |
| `dm_control/manipulator-insert_peg-v0`  | Planar manipulator challenge of inserting a peg into a slot.                                                            | 5          | 44              |
| `dm_control/pendulum-swingup-v0`        | Classic inverted pendulum task with a torque-limited actuator, requiring multiple swings to swing up and balance.       | 1          | 3               |
| `dm_control/point_mass-easy-v0`         | Planar point mass in an easy task, with actuators corresponding to global x and y axes                                  | 2          | 4               |
| `dm_control/point_mass-hard-v0`         | Planar point mass in a hard task, with randomized control gains per episode, challenging memoryless agents.             | 2          | 4               |
| `dm_control/quadruped-walk-v0`          | Four-legged robot (Quadruped) tasked with walking.                                                                      | 12         | 78              |
| `dm_control/quadruped-run-v0`           | Quadruped robot required to run, balancing speed and stability.                                                         | 12         | 78              |
| `dm_control/quadruped-escape-v0`        | Quadruped robot in an escape task, combining locomotion with environmental interaction.                                 | 12         | 101             |
| `dm_control/quadruped-fetch-v0`         | Quadruped robot playing fetch, involving complex movements and object interaction.                                      | 12         | 90              |
| `dm_control/reacher-easy-v0`            | Two-link planar reacher with a randomized target, easier version with a larger target sphere.                           | 2          | 6               |
| `dm_control/reacher-hard-v0`            | Similar to `reacher-easy-v0`, but with a smaller target sphere for increased difficulty.                                | 2          | 6               |
| `dm_control/stacker-stack_2-v0`         | Stacker task requiring stacking of 2 boxes, with rewards for correct placement and gripper positioning.                 | 5          | 49              |
| `dm_control/stacker-stack_4-v0`         | More complex stacker task with 4 boxes, increasing the stacking challenge.                                              | 5          | 63              |
| `dm_control/swimmer-swimmer6-v0`        | Six-link planar swimmer in fluid dynamics, rewarded for positioning the nose inside a target.                           | 5          | 25              |
| `dm_control/swimmer-swimmer15-v0`       | Fifteen-link planar swimmer, a more complex version of the swimmer task with extended dynamics.                         | 14         | 61              |
| `dm_control/walker-stand-v0`            | Planar walker robot tasked with maintaining an upright torso and minimal height.                                        | 6          | 24              |
| `dm_control/walker-walk-v0`             | Planar walker robot walking task, focusing on forward velocity and stability.                                           | 6          | 24              |
| `dm_control/walker-run-v0`              | Planar walker robot running task, aiming for high speed and balance.                                                    | 6          | 24              |

## MP Environments

[//]: <> (These environments are wrapped-versions of their Deep Mind Control Suite &#40;DMC&#41; counterparts. Given most task can be)
[//]: <> (solved in shorter horizon lengths than the original 1000 steps, we often shorten the episodes for those task.)

| Name                                     | Description                                                                    | Trajectory Horizon | Action Dimension | Context Dimension |
| ---------------------------------------- | ------------------------------------------------------------------------------ | ------------------ | ---------------- | ----------------- |
| `dm_control_ProDMP/ball_in_cup-catch-v0` | A ProMP wrapped version of the "catch" task for the "ball_in_cup" environment. | 1000               | 10               | 2                 |
| `dm_control_DMP/ball_in_cup-catch-v0`    | A DMP wrapped version of the "catch" task for the "ball_in_cup" environment.   | 1000               | 10               | 2                 |
| `dm_control_ProDMP/reacher-easy-v0`      | A ProMP wrapped version of the "easy" task for the "reacher" environment.      | 1000               | 10               | 4                 |
| `dm_control_DMP/reacher-easy-v0`         | A DMP wrapped version of the "easy" task for the "reacher" environment.        | 1000               | 10               | 4                 |
| `dm_control_ProDMP/reacher-hard-v0`      | A ProMP wrapped version of the "hard" task for the "reacher" environment.      | 1000               | 10               | 4                 |
| `dm_control_DMP/reacher-hard-v0`         | A DMP wrapped version of the "hard" task for the "reacher" environment.        | 1000               | 10               | 4                 |
