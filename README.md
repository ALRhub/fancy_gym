## ALR Custom Environments
    
This repository collects custom RL envs not included in Suits like OpenAI gym, rllab, etc. 
Creating a custom (Mujoco) gym environement can be done according to this guide: https://github.com/openai/gym/blob/master/docs/creating-environments.md

## Environments
Currently we have the following environements: 

### Mujoco

|Name| Description|
|---|---|
|`ALRReacher-v0`|modification (5 links) of Mujoco Gym's Reacher (2 links)|
|`ALRReacherSparse-v0`|Same as `ALRReacher-v0`, but the distance penalty is only provided in the last time step.|
    
### Classic Control

|Name| Description|
|---|---|
|`SimpleReacher-v0`| Simple Reaching Task without any physics simulation. Returns no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions towards the end of the trajectory.|

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
