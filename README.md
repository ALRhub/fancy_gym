## ALR Custom Environments
    
This repository collects custom RL envs not included in Suits like OpenAI gym, rllab, etc. 
Creating a custom (Mujoco) gym environement can be done according to [this guide](https://github.com/openai/gym/blob/master/docs/creating-environments.md).
For stochastic search problems with gym interace use the Rosenbrock reference implementation.

## Environments
Currently we have the following environements: 

### Mujoco

|Name| Description|
|---|---|
|`ALRReacher-v0`|Modified (5 links) Mujoco gym's `Reacher-v2` (2 links)|
|`ALRReacherSparse-v0`|Same as `ALRReacher-v0`, but the distance penalty is only provided in the last time step.|
|`ALRReacherSparseBalanced-v0`|Same as `ALRReacherSparse-v0`, but the the end effector has to stay upright.|
|`ALRReacherShort-v0`|Same as `ALRReacher-v0`, but the episode length is reduced to 50.|
|`ALRReacherShortSparse-v0`|Combination of `ALRReacherSparse-v0` and `ALRReacherShort-v0`.|
|`ALRReacher7-v0`|Modified (7 links) Mujoco gym's `Reacher-v2` (2 links)|
|`ALRReacher7Sparse-v0`|Same as `ALRReacher7-v0`, but the distance penalty is only provided in the last time step.|
    
### Classic Control

|Name| Description|
|---|---|
|`SimpleReacher-v0`| Simple Reaching Task without any physics simulation. Returns no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions towards the end of the trajectory.|

### Stochastic Search
|Name| Description|
|---|---|
|`Rosenbrock{dim}-v0`| Gym interface for Rosenbrock function. `{dim}` is one of 5, 10, 25, 50 or 100. | 


## INSTALL
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
4. Use (see [example.py](./example.py)): 
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
