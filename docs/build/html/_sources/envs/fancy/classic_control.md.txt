# Classic Control

Classic Control environments provide a foundational platform for exploring and experimenting with RL algorithms. These environments are designed to be simple, allowing researchers and practitioners to focus on the fundamental principles of control without the complexities of high-dimensional and physics-based simulations.

## Step-Based Environments

| Name                         | Description                                                                                                                                                                                                         | Horizon | Action Dimension | Observation Dimension |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---------------- | --------------------- |
| `fancy/SimpleReacher-v0`     | Simple reaching task (2 links) without any physics simulation. Provides no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions towards the end of the trajectory. | 200     | 2                | 9                     |
| `fancy/LongSimpleReacher-v0` | Simple reaching task (5 links) without any physics simulation. Provides no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions towards the end of the trajectory. | 200     | 5                | 18                    |
| `fancy/ViaPointReacher-v0`   | Simple reaching task leveraging a via point, which supports self collision detection. Provides a reward only at 100 and 199 for reaching the viapoint and goal point, respectively.                                 | 200     | 5                | 18                    |
| `fancy/HoleReacher-v0`       | 5 link reaching task where the end-effector needs to reach into a narrow hole without collding with itself or walls.                                                                                                | 200     | 5                | 18                    |

## MP Environments

| Name                                | Description                                                                                              | Horizon | Action Dimension | Context Dimension |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------- | ------- | ---------------- | ----------------- |
| `fancy_DMP/ViaPointReacher-v0`      | A DMP provides a trajectory for the `fancy/ViaPointReacher-v0` task.                                     | 200     | 25               |
| `fancy_DMP/HoleReacherFixedGoal-v0` | A DMP provides a trajectory for the `fancy/HoleReacher-v0` task with a fixed goal attractor.             | 200     | 25               |
| `fancy_DMP/HoleReacher-v0`          | A DMP provides a trajectory for the `fancy/HoleReacher-v0` task. The goal attractor needs to be learned. | 200     | 30               |

[//]: |`fancy/HoleReacherProMPP-v0`|
