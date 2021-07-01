## ALR Environments
    
This repository collects custom Robotics environments not included in benchmark suites like OpenAI gym, rllab, etc. 
Creating a custom (Mujoco) gym environment can be done according to [this guide](https://github.com/openai/gym/blob/master/docs/creating-environments.md).
For stochastic search problems with gym interface use the `Rosenbrock-v0` reference implementation.
We also support to solve environments with DMPs. When adding new DMP tasks check the `ViaPointReacherDMP-v0` reference implementation.
When simply using the tasks, you can also leverage the wrapper class `DmpWrapper` to turn normal gym environments in to DMP tasks.

## Environments
Currently we have the following environments: 

### Mujoco

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
|`ALRBallInACupGoal-v0`| Similiar to `ALRBallInACupSimple-v0` but the ball needs to be caught at a specified goal position | 4000 | 7 | wip
    
### Classic Control

|Name| Description|Horizon|Action Dimension|Observation Dimension
|---|---|---|---|---|
|`SimpleReacher-v0`| Simple reaching task (2 links) without any physics simulation. Provides no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions towards the end of the trajectory.| 200 | 2 | 9
|`LongSimpleReacher-v0`| Simple reaching task (5 links) without any physics simulation. Provides no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions towards the end of the trajectory.| 200 | 5 | 18
|`ViaPointReacher-v0`| Simple reaching task leveraging a via point, which supports self collision detection. Provides a reward only at 100 and 199 for reaching the viapoint and goal point, respectively.| 200 | 5 | 18 
|`HoleReacher-v0`| 5 link reaching task where the end-effector needs to reach into a narrow hole without collding with itself or walls | 200 | 5 | 18

### DMP Environments
These environments are closer to stochastic search. They always execute a full trajectory, which is computed by a DMP and executed by a controller, e.g. a PD controller.
The goal is to learn the parameters of this DMP to generate a suitable trajectory. 
All environments provide the full episode reward and additional information about early terminations, e.g. due to collisions. 

|Name| Description|Horizon|Action Dimension|Context Dimension
|---|---|---|---|---|
|`ViaPointReacherDMP-v0`| A DMP provides a trajectory for the `ViaPointReacher-v0` task. | 200 | 25
|`HoleReacherFixedGoalDMP-v0`| A DMP provides a trajectory for the `HoleReacher-v0` task with a fixed goal attractor. | 200 | 25
|`HoleReacherDMP-v0`| A DMP provides a trajectory for the `HoleReacher-v0` task. The goal attractor needs to be learned. | 200 | 30 
|`ALRBallInACupSimpleDMP-v0`| A DMP provides a trajectory for the `ALRBallInACupSimple-v0` task where only 3 joints are actuated. | 4000 | 15
|`ALRBallInACupDMP-v0`| A DMP provides a trajectory for the `ALRBallInACup-v0` task. | 4000 | 35
|`ALRBallInACupGoalDMP-v0`| A DMP provides a trajectory for the `ALRBallInACupGoal-v0` task. | 4000 | 35 | 3 

[//]:  |`HoleReacherDetPMP-v0`|

### OpenAi-gym Environments
These environments are wrapped-versions of their OpenAi-gym counterparts.

|Name| Description|Horizon|Action Dimension|Context Dimension
|---|---|---|---|---|
|`ContinuousMountainCarDetPMP-v0`| A DetPmP wrapped version of the ContinuousMountainCar-v0 environment. | 100 | 1
|`ReacherDetPMP-v2`| A DetPmP wrapped version of the Reacher-v2 environment. | 50 | 2
|`FetchSlideDenseDetPMP-v1`| A DetPmP wrapped version of the FetchSlideDense-v1 environment. | 50 | 4 
|`FetchReachDenseDetPMP-v1`| A DetPmP wrapped version of the FetchReachDense-v1 environment. | 50 | 4


### Stochastic Search
|Name| Description|Horizon|Action Dimension|Observation Dimension
|---|---|---|---|---|
|`Rosenbrock{dim}-v0`| Gym interface for Rosenbrock function. `{dim}` is one of 5, 10, 25, 50 or 100. | 1 | `{dim}` | 0


## Install
1. Clone the repository 
```bash 
git clone git@github.com:ALRhub/alr_envs.git
```
2. Go to the folder 
```bash 
cd alr_envs
```
3. Install with 
```bash 
pip install -e . 
```
4. Use (see [example.py](alr_envs/examples/examples_general.py)): 
```python
import gym

env = gym.make('alr_envs:SimpleReacher-v0')
state = env.reset()

for i in range(10000):
    state, reward, done, info = env.step(env.action_space.sample())
    if i % 5 == 0:
        env.render()

    if done:
        state = env.reset()

``` 

For an example using a DMP wrapped env and asynchronous sampling look at [mp_env_async_sampler.py](./alr_envs/utils/mp_env_async_sampler.py)