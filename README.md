<h1 align="center">
  <br>
  <img src='./icon.svg' width="250px">
  <br><br>
  <b>Fancy Gym</b>
  <br>
  <br>
</h1>

| :exclamation:  Fancy Gym has recently received a mayor refactor, which also updated many of the used dependencies to current versions. The update has brought some breaking changes. If you want to access the old version, check out the legacy branch. Find out more about what changed [here](TODO). |
| ------------------------------------------------------------ |

Built upon the foundation of [Gymnasium](https://gymnasium.farama.org/) (a maintained fork of OpenAI’s renowned Gym library) `fancy_gym` offers a comprehensive collection of reinforcement learning environments.

**Key Features**:

- **New Challenging Environments**: We've introduced several new environments (Panda Box Pushing, Table Tennis, etc.) that present a higher degree of difficulty, pushing the boundaries of reinforcement learning research.
- **Support for Movement Primitives**: `fancy_gym` supports a range of movement primitives (MPs), including Dynamic Movement Primitives (DMPs), Probabilistic Movement Primitives (ProMP), and Probabilistic Dynamic Movement Primitives (ProDMP).
- **Upgrade to Movement Primitives**: With our framework, it's straightforward to transform standard Gymnasium environments into environments that support movement primitives.
- **Benchmark Suite Compatibility**: `fancy_gym` makes it easy to access renowned benchmark suites such as [DeepMind Control](https://deepmind.com/research/publications/2020/dm-control-Software-and-Tasks-for-Continuous-Control) and [Metaworld](https://meta-world.github.io/), whether you want to use them in the normal step-based or a MP-based setting.
- **Contribute Your Own Environments**: If you're inspired to create custom gym environments, both step-based and with movement primitives, this [guide](https://www.gymlibrary.dev/content/environment_creation/) will assist you. We encourage and highly appreciate submissions via PRs to integrate these environments into `fancy_gym`.

## Movement Primitive Environments (Episode-Based/Black-Box Environments)

Movement primitive (MP) environments differ from traditional step-based environments. They align more with concepts from stochastic search, black-box optimization, and methods commonly found in classical robotics and control. Instead of individual steps, MP environments operate on an episode basis, executing complete trajectories. These trajectories are produced by trajectory generators like Dynamic Movement Primitives (DMP), Probabilistic Movement Primitives (ProMP) or Probabilistic Dynamic Movement Primitives (ProDMP).

Once generated, these trajectories are converted into step-by-step actions using a trajectory tracking controller. The specific controller chosen depends on the environment's requirements. Currently, we support position, velocity, and PD-Controllers tailored for position, velocity, and torque control. Additionally, we have a specialized controller designed for the MetaWorld control suite.

While the overarching objective of MP environments remains the learning of an optimal policy, the actions here represent the parametrization of motion primitives to craft the right trajectory. Our framework further enhances this by accommodating a contextual setting. At the episode's onset, we present the context space—a subset of the observation space. This demands the prediction of a new action or MP parametrization for every unique context.

## Installation

1. Clone the repository

```bash 
git clone git@github.com:ALRhub/fancy_gym.git
```

2. Go to the folder

```bash 
cd fancy_gym
```

3. Install with

```bash 
pip install -e .
```

We have a few optional dependencies. CHeck them out in the setup.py or just install all of them via

```bash 
pip install -e '.[all]'
```


## How to use Fancy Gym

We will only show the basics here and prepared [multiple examples](fancy_gym/examples/) for a more detailed look.

### Step-Based Environments
Regular step based environments added by Fancy Gym are added into the ```fancy/``` namespace.

| :exclamation:  Legacy versions of Fancy Gym used ```fancy_gym.make(...)```. This is no longer supported and will raise an Exception on new versions. |
| ------------------------------------------------------------ |

```python
import gymnasium as gym
import fancy_gym

env = gym.make('fancy/Reacher5d-v0')
observation = env.reset(seed=1)

for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if i % 5 == 0:
        env.render()

    if terminated or truncated:
        observation = env.reset()
```

### Black-box Environments

All environments provide by default the cumulative episode reward, this can however be changed if necessary. Optionally, each environment returns all collected information from each step as part of the infos. This information is, however, mainly meant for debugging as well as logging and not for training.

|Key| Description|Type
|---|---|---|
`positions`| Generated trajectory from MP | Optional
`velocities`| Generated trajectory from MP | Optional
`step_actions`| Step-wise executed action based on controller output | Optional
`step_observations`| Step-wise intermediate observations | Optional
`step_rewards`| Step-wise rewards | Optional
`trajectory_length`| Total number of environment interactions | Always
`other`| All other information from the underlying environment are returned as a list with length `trajectory_length` maintaining the original key. In case some information are not provided every time step, the missing values are filled with `None`. | Always

Existing MP tasks can be created the same way as above. The namespace of a MP-variant of an environment is given by ```<original namespace>_<MP name>/```.
Just keep in mind, calling `step()` executes a full trajectory.

> **Note:**   
> Currently, we are also in the process of enabling replanning as well as learning of sub-trajectories.
> This allows to split the episode into multiple trajectories and is a hybrid setting between step-based and
> black-box leaning.
> While this is already implemented, it is still in beta and requires further testing.
> Feel free to try it and open an issue with any problems that occur.

```python
import gymnasium as gym
import fancy_gym

env = gym.make('fancy_ProMP/Reacher5d-v0')
# or env = fancy_gym.make('metaworld_ProDMP/reach-v2')
# or env = fancy_gym.make('dm_control_DMP/ball_in_cup-catch-v0')

# render() can be called once in the beginning with all necessary arguments.
# To turn it of again just call render() without any arguments. 
env.render(mode='human')

# This returns the context information, not the full state observation
observation = env.reset(seed=1)

for i in range(5):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # Done is always True as we are working on the episode level, hence we always reset()
    observation = env.reset()
```

To show all available environments, we provide some additional convenience variables. All of them return a dictionary
with two keys `DMP` and `ProMP` that store a list of available environment ids.

```python
import fancy_gym

print("All Black-box tasks:")
print(fancy_gym.ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS)

print("Fancy Black-box tasks:")
print(fancy_gym.ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS)

print("OpenAI Gym Black-box tasks:")
print(fancy_gym.ALL_GYM_MOVEMENT_PRIMITIVE_ENVIRONMENTS)

print("Deepmind Control Black-box tasks:")
print(fancy_gym.ALL_DMC_MOVEMENT_PRIMITIVE_ENVIRONMENTS)

print("MetaWorld Black-box tasks:")
print(fancy_gym.ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS)
```

### How to create a new MP task

In case a required task is not supported yet in the MP framework, it can be created relatively easy. For the task at
hand, the following [interface](fancy_gym/black_box/raw_interface_wrapper.py) needs to be implemented.

```python
from abc import abstractmethod
from typing import Union, Tuple

import gymnasium as gym
import numpy as np


class RawInterfaceWrapper(gym.Wrapper):
    mp_config = { # Default configurations for MPs can be ovveritten by defining them here.
        'ProMP': {},
        'DMP': {},
        'ProDMP': {},
    }    
    
    @property
    def context_mask(self) -> np.ndarray:
        """
        Returns boolean mask of the same shape as the observation space.
        It determines whether the observation is returned for the contextual case or not.
        This effectively allows to filter unwanted or unnecessary observations from the full step-based case.
        E.g. Velocities starting at 0 are only changing after the first action. Given we only receive the 
        context/part of the first observation, the velocities are not necessary in the observation for the task.
        Returns:
            bool array representing the indices of the observations

        """
        return np.ones(self.env.observation_space.shape[0], dtype=bool)

    @property
    @abstractmethod
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        """
            Returns the current position of the action/control dimension.
            The dimensionality has to match the action/control dimension.
            This is not required when exclusively using velocity control,
            it should, however, be implemented regardless.
            E.g. The joint positions that are directly or indirectly controlled by the action.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        """
            Returns the current velocity of the action/control dimension.
            The dimensionality has to match the action/control dimension.
            This is not required when exclusively using position control,
            it should, however, be implemented regardless.
            E.g. The joint velocities that are directly or indirectly controlled by the action.
        """
        raise NotImplementedError()

```

If you created a new task wrapper, feel free to open a PR, so we can integrate it for others to use as well. Without the
integration the task can still be used. A rough outline can be shown here, for more details we recommend having a look
at the [examples](fancy_gym/examples/).

If the step-based is already registered with gym, you can simply do the following:

```python
fancy_gym.upgrade(
    id='custom/cool_new_env-v0',
    mp_wrapper=my_custom_MPWrapper
)
```

If the step-based is not yet registered with gym we can add both the step-based and MP-versions via

```python
fancy_gym.register(
    id='custom/cool_new_env-v0',
    entry_point=my_custom_env,
    mp_wrapper=my_custom_MPWrapper
)
```

From this point on, you can access MP-version of your environments via

```python
env = gym.make('custom_ProDMP/cool_new_env-v0')

rewards = 0
observation = env.reset()

# number of samples/full trajectories (multiple environment steps)
for i in range(5):
    ac = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(ac)
    rewards += reward

    if terminated or truncated:
        print(base_env_id, rewards)
        rewards = 0
        observation = env.reset()
```

## Icon Attribution
The icon is based on the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) icon as can be found [here](https://gymnasium.farama.org/_static/img/gymnasium_black.svg).
