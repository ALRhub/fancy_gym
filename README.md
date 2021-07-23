## ALR Robotics Control Environments
    
This repository collects custom Robotics environments not included in benchmark suites like OpenAI gym, rllab, etc. 
Creating a custom (Mujoco) gym environment can be done according to [this guide](https://github.com/openai/gym/blob/master/docs/creating-environments.md).
For stochastic search problems with gym interface use the `Rosenbrock-v0` reference implementation.
We also support to solve environments with Dynamic Movement Primitives (DMPs) and Probabilistic Movement Primitives (DetPMP, we only consider the mean usually). 

## Step-based Environments
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

## Motion Primitive Environments (Episodic environments)
Unlike step-based environments, these motion primitive (MP) environments are closer to stochastic search and what can be found in robotics. They always execute a full trajectory, which is computed by a Dynamic Motion Primitive (DMP) or Probabilitic Motion Primitive (DetPMP) and translated into individual actions with a controller, e.g. a PD controller. The actual Controller, however, depends on the type of environment, i.e. position, velocity, or torque controlled.
The goal is to learn the parametrization of the motion primitives in order to generate a suitable trajectory. 
MP This can also be done in a contextual setting, where all changing elements of the task are exposed once in the beginning. This requires to find a new parametrization for each trajectory.
All environments provide the full cumulative episode reward and additional information about early terminations, e.g. due to collisions. 

### Classic Control
|Name| Description|Horizon|Action Dimension|Context Dimension
|---|---|---|---|---|
|`ViaPointReacherDMP-v0`| A DMP provides a trajectory for the `ViaPointReacher-v0` task. | 200 | 25
|`HoleReacherFixedGoalDMP-v0`| A DMP provides a trajectory for the `HoleReacher-v0` task with a fixed goal attractor. | 200 | 25
|`HoleReacherDMP-v0`| A DMP provides a trajectory for the `HoleReacher-v0` task. The goal attractor needs to be learned. | 200 | 30 
|`ALRBallInACupSimpleDMP-v0`| A DMP provides a trajectory for the `ALRBallInACupSimple-v0` task where only 3 joints are actuated. | 4000 | 15
|`ALRBallInACupDMP-v0`| A DMP provides a trajectory for the `ALRBallInACup-v0` task. | 4000 | 35
|`ALRBallInACupGoalDMP-v0`| A DMP provides a trajectory for the `ALRBallInACupGoal-v0` task. | 4000 | 35 | 3 

[//]:  |`HoleReacherDetPMP-v0`|

### OpenAI gym Environments
These environments are wrapped-versions of their OpenAI-gym counterparts.

|Name| Description|Trajectory Horizon|Action Dimension|Context Dimension
|---|---|---|---|---|
|`ContinuousMountainCarDetPMP-v0`| A DetPmP wrapped version of the ContinuousMountainCar-v0 environment. | 100 | 1
|`ReacherDetPMP-v2`| A DetPmP wrapped version of the Reacher-v2 environment. | 50 | 2
|`FetchSlideDenseDetPMP-v1`| A DetPmP wrapped version of the FetchSlideDense-v1 environment. | 50 | 4 
|`FetchReachDenseDetPMP-v1`| A DetPmP wrapped version of the FetchReachDense-v1 environment. | 50 | 4

### Deep Mind Control Suite Environments
These environments are wrapped-versions of their Deep Mind Control Suite (DMC) counterparts.
Given most task can be solved in shorter horizon lengths than the original 1000 steps, we often shorten the episodes for those task. 

|Name| Description|Trajectory Horizon|Action Dimension|Context Dimension
|---|---|---|---|---|
|`dmc_ball_in_cup-catch_detpmp-v0`| A DetPmP wrapped version of the "catch" task for the "ball_in_cup" environment. | 50 | 10 | 2
|`dmc_ball_in_cup-catch_dmp-v0`| A DMP wrapped version of the "catch" task for the "ball_in_cup" environment. | 50| 10 | 2
|`dmc_reacher-easy_detpmp-v0`| A DetPmP wrapped version of the "easy" task for the "reacher" environment. | 1000 | 10 | 4
|`dmc_reacher-easy_dmp-v0`| A DMP wrapped version of the "easy" task for the "reacher" environment. | 1000| 10 | 4
|`dmc_reacher-hard_detpmp-v0`| A DetPmP wrapped version of the "hard" task for the "reacher" environment.| 1000 | 10 | 4
|`dmc_reacher-hard_dmp-v0`| A DMP wrapped version of the "hard" task for the "reacher" environment. | 1000 | 10 | 4


## Stochastic Search
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
