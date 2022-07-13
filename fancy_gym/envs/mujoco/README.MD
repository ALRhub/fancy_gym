# Custom Mujoco tasks

## Step-based Environments
|Name| Description|Horizon|Action Dimension|Observation Dimension
|---|---|---|---|---|
|`ALRReacher-v0`|Modified (5 links) Mujoco gym's `Reacher-v2` (2 links)| 200 | 5 | 21
|`ALRReacherSparse-v0`|Same as `ALRReacher-v0`, but the distance penalty is only provided in the last time step.| 200 | 5 | 21
|`ALRReacherSparseBalanced-v0`|Same as `ALRReacherSparse-v0`, but the end-effector has to remain upright.| 200 | 5 | 21
|`ALRLongReacher-v0`|Modified (7 links) Mujoco gym's `Reacher-v2` (2 links)| 200 | 7 | 27
|`ALRLongReacherSparse-v0`|Same as `ALRLongReacher-v0`, but the distance penalty is only provided in the last time step.| 200 | 7 | 27
|`ALRLongReacherSparseBalanced-v0`|Same as `ALRLongReacherSparse-v0`, but the end-effector has to remain upright.| 200 | 7 | 27
|`ALRBallInACupSimple-v0`| Ball-in-a-cup task where a robot needs to catch a ball attached to a cup at its end-effector. | 4000 | 3 | wip
|`ALRBallInACup-v0`| Ball-in-a-cup task where a robot needs to catch a ball attached to a cup at its end-effector | 4000 | 7 | wip
|`ALRBallInACupGoal-v0`| Similar to `ALRBallInACupSimple-v0` but the ball needs to be caught at a specified goal position | 4000 | 7 | wip
    