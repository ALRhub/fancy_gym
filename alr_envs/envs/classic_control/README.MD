### Classic Control

## Step-based Environments
|Name| Description|Horizon|Action Dimension|Observation Dimension
|---|---|---|---|---|
|`SimpleReacher-v0`| Simple reaching task (2 links) without any physics simulation. Provides no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions towards the end of the trajectory.| 200 | 2 | 9
|`LongSimpleReacher-v0`| Simple reaching task (5 links) without any physics simulation. Provides no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions towards the end of the trajectory.| 200 | 5 | 18
|`ViaPointReacher-v0`| Simple reaching task leveraging a via point, which supports self collision detection. Provides a reward only at 100 and 199 for reaching the viapoint and goal point, respectively.| 200 | 5 | 18 
|`HoleReacher-v0`| 5 link reaching task where the end-effector needs to reach into a narrow hole without collding with itself or walls | 200 | 5 | 18

## MP Environments
|Name| Description|Horizon|Action Dimension|Context Dimension
|---|---|---|---|---|
|`ViaPointReacherDMP-v0`| A DMP provides a trajectory for the `ViaPointReacher-v0` task. | 200 | 25
|`HoleReacherFixedGoalDMP-v0`| A DMP provides a trajectory for the `HoleReacher-v0` task with a fixed goal attractor. | 200 | 25
|`HoleReacherDMP-v0`| A DMP provides a trajectory for the `HoleReacher-v0` task. The goal attractor needs to be learned. | 200 | 30

[//]:  |`HoleReacherProMPP-v0`|