# Mujoco

## Step-based Environments

### Environments made by Fancy Gym

#### Beer Pong

TODO: Change image

<div class='center'>
   <img src="../../_static/imgs/Box_Pushing.gif" style="margin: 5%; width: 45%;">
</div>
<br>

| Name                            | Description                                                                                    | Horizon | Action Dimension | Observation Dimension |
| ------------------------------- | ---------------------------------------------------------------------------------------------- | ------- | ---------------- | --------------------- |
| `fancy/BeerPong-v0`             | Beer Pong task, based on a custom environment with multiple task variations                    | 300     | 3                | 29                    |
| `fancy/BeerPongStepBased-v0`    | Step-based rewards for the Beer Pong task, based on a custom environment with episodic rewards | 300     | 3                | 29                    |
| `fancy/BeerPongFixedRelease-v0` | Beer Pong with fixed release, based on a custom environment with episodic rewards              | 300     | 3                | 29                    |

#### Box Pushing

<div class='center'>
   <img src="../../_static/imgs/Box_Pushing.gif" style="margin: 5%; width: 45%;">
</div>
<br>

| Name                                       | Description                                                          | Horizon | Action Dimension | Observation Dimension |
| ------------------------------------------ | -------------------------------------------------------------------- | ------- | ---------------- | --------------------- |
| `fancy/BoxPushingDense-v0`                 | Custom Box-pushing task with dense rewards                           | 100     | 3                | 13                    |
| `fancy/BoxPushingTemporalSparse-v0`        | Custom Box-pushing task with temporally sparse rewards               | 100     | 3                | 13                    |
| `fancy/BoxPushingTemporalSpatialSparse-v0` | Custom Box-pushing task with temporally and spatially sparse rewards | 100     | 3                | 13                    |

#### Table Tennis

TODO: Change image

<div class='center'>
   <img src="../../_static/imgs/Box_Pushing.gif" style="margin: 5%; width: 45%;">
</div>
<br>

| Name                                | Description                                                                                        | Horizon | Action Dimension | Observation Dimension |
| ----------------------------------- | -------------------------------------------------------------------------------------------------- | ------- | ---------------- | --------------------- |
| `fancy/TableTennis2D-v0`            | Table Tennis task with 2D context, based on a custom environment for table tennis                  | 350     | 7                | 19                    |
| `fancy/TableTennis2DReplan-v0`      | Table Tennis task with 2D context and replanning, based on a custom environment for table tennis   | 350     | 7                | 19                    |
| `fancy/TableTennis4D-v0`            | Table Tennis task with 4D context, based on a custom environment for table tennis                  | 350     | 7                | 22                    |
| `fancy/TableTennis4DReplan-v0`      | Table Tennis task with 4D context and replanning, based on a custom environment for table tennis   | 350     | 7                | 22                    |
| `fancy/TableTennisWind-v0`          | Table Tennis task with wind effects, based on a custom environment for table tennis                | 350     | 7                | 19                    |
| `fancy/TableTennisGoalSwitching-v0` | Table Tennis task with goal switching, based on a custom environment for table tennis              | 350     | 7                | 19                    |
| `fancy/TableTennisWindReplan-v0`    | Table Tennis task with wind effects and replanning, based on a custom environment for table tennis | 350     | 7                | 19                    |

### Variations of existing environments

| Name                                 | Description                                                                                      | Horizon | Action Dimension | Observation Dimension |
| ------------------------------------ | ------------------------------------------------------------------------------------------------ | ------- | ---------------- | --------------------- |
| `fancy/Reacher-v0`                   | Modified (5 links) gymnasiums's mujoco `Reacher-v2` (2 links)                                    | 200     | 5                | 21                    |
| `fancy/ReacherSparse-v0`             | Same as `fancy/Reacher-v0`, but the distance penalty is only provided in the last time step.     | 200     | 5                | 21                    |
| `fancy/ReacherSparseBalanced-v0`     | Same as `fancy/ReacherSparse-v0`, but the end-effector has to remain upright.                    | 200     | 5                | 21                    |
| `fancy/LongReacher-v0`               | Modified (7 links) gymnasiums's mujoco `Reacher-v2` (2 links)                                    | 200     | 7                | 27                    |
| `fancy/LongReacherSparse-v0`         | Same as `fancy/LongReacher-v0`, but the distance penalty is only provided in the last time step. | 200     | 7                | 27                    |
| `fancy/LongReacherSparseBalanced-v0` | Same as `fancy/LongReacherSparse-v0`, but the end-effector has to remain upright.                | 200     | 7                | 27                    |
| `fancy/Reacher5d-v0`                 | Reacher task with 5 links, based on Gymnasium's `gym.envs.mujoco.ReacherEnv`                     | 200     | 5                | 20                    |
| `fancy/Reacher5dSparse-v0`           | Sparse Reacher task with 5 links, based on Gymnasium's `gym.envs.mujoco.ReacherEnv`              | 200     | 5                | 20                    |
| `fancy/Reacher7d-v0`                 | Reacher task with 7 links, based on Gymnasium's `gym.envs.mujoco.ReacherEnv`                     | 200     | 7                | 22                    |
| `fancy/Reacher7dSparse-v0`           | Sparse Reacher task with 7 links, based on Gymnasium's `gym.envs.mujoco.ReacherEnv`              | 200     | 7                | 22                    |
| `fancy/HopperJumpSparse-v0`          | Hopper Jump task with sparse rewards, based on Gymnasium's `gym.envs.mujoco.Hopper`              | 250     | 3                | 15 / 16\*             |
| `fancy/HopperJump-v0`                | Hopper Jump task with continuous rewards, based on Gymnasium's `gym.envs.mujoco.Hopper`          | 250     | 3                | 15 / 16\*             |
| `fancy/AntJump-v0`                   | Ant Jump task, based on Gymnasium's `gym.envs.mujoco.Ant`                                        | 200     | 8                | 119                   |
| `fancy/HalfCheetahJump-v0`           | HalfCheetah Jump task, based on Gymnasium's `gym.envs.mujoco.HalfCheetah`                        | 100     | 6                | 112                   |
| `fancy/HopperJumpOnBox-v0`           | Hopper Jump on Box task, based on Gymnasium's `gym.envs.mujoco.Hopper`                           | 250     | 4                | 16 / 100\*            |
| `fancy/HopperThrow-v0`               | Hopper Throw task, based on Gymnasium's `gym.envs.mujoco.Hopper`                                 | 250     | 3                | 18 / 100\*            |
| `fancy/HopperThrowInBasket-v0`       | Hopper Throw in Basket task, based on Gymnasium's `gym.envs.mujoco.Hopper`                       | 250     | 3                | 18 / 100\*            |
| `fancy/Walker2DJump-v0`              | Walker 2D Jump task, based on Gymnasium's `gym.envs.mujoco.Walker2d`                             | 300     | 6                | 18 / 19\*             |

\*Observation dimensions depend on configuration.

<!--
No longer used?
| Name                        | Description                                                                                         | Horizon | Action Dimension | Observation Dimension |
| --------------------------- | --------------------------------------------------------------------------------------------------- | ------- | ---------------- | --------------------- |
| `fancy/BallInACupSimple-v0` | Ball-in-a-cup task where a robot needs to catch a ball attached to a cup at its end-effector.       | 4000    | 3                | wip                   |
| `fancy/BallInACup-v0`       | Ball-in-a-cup task where a robot needs to catch a ball attached to a cup at its end-effector        | 4000    | 7                | wip                   |
| `fancy/BallInACupGoal-v0`   | Similar to `fancy/BallInACupSimple-v0` but the ball needs to be caught at a specified goal position | 4000    | 7                | wip                   |
-->

## MP Environments

Many of these envs also exist as MP-variants. Refer to them using `fancy_DMP/<name>` `fancy_ProMP/<name>` or `fancy_ProDMP/<name>`.
