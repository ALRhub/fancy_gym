<h1 align="center">
  <br>
  <img src='https://raw.githubusercontent.com/ALRhub/fancy_gym/master/icon.svg' width="250px">
  <br><br>
  <b>Fancy Gym</b>
  <br><br>
</h1>

Built upon the foundation of [Gymnasium](https://gymnasium.farama.org/) (a maintained fork of OpenAI’s renowned Gym library) `fancy_gym` offers a comprehensive collection of reinforcement learning environments.

**Key Features**:

- **New Challenging Environments**: `fancy_gym` includes several new environments (Panda Box Pushing, Table Tennis, etc.) that present a higher degree of difficulty, pushing the boundaries of reinforcement learning research.
- **Support for Movement Primitives**: `fancy_gym` supports a range of movement primitives (MPs), including Dynamic Movement Primitives (DMPs), Probabilistic Movement Primitives (ProMP), and Probabilistic Dynamic Movement Primitives (ProDMP).
- **Upgrade to Movement Primitives**: With our framework, it's straightforward to transform standard Gymnasium environments into environments that support movement primitives.
- **Benchmark Suite Compatibility**: `fancy_gym` makes it easy to access renowned benchmark suites such as [DeepMind Control](https://deepmind.com/research/publications/2020/dm-control-Software-and-Tasks-for-Continuous-Control) and [Metaworld](https://meta-world.github.io/), whether you want to use them in the regular step-based setting or using MPs.
- **Contribute Your Own Environments**: If you're inspired to create custom gym environments, both step-based and with movement primitives, this [guide](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) will assist you. We encourage and highly appreciate submissions via PRs to integrate these environments into `fancy_gym`.

## Movement Primitive Environments (Episode-Based/Black-Box Environments)

<p align="justify">
In step-based environments, actions are determined step by step, with each individual observation directly mapped to a corresponding action. Contrary to this, in episodic MP-based (Movement Primitive-based) environments, the process is different. Here, rather than responding to individual observations, a broader context is considered at the start of each episode. This context is used to define parameters for Movement Primitives (MPs), which then describe a complete trajectory. The trajectory is executed over the entire episode using a tracking controller, allowing for the enactment of complex, continuous sequences of actions. This approach contrasts with the discrete, moment-to-moment decision-making of step-based environments and integrates concepts from stochastic search and black-box optimization, commonly found in classical robotics and control.
</p>

For a more extensive explaination, please have a look at our Documentation-TODO:Link.

## Installation

We recommend installing `fancy_gym` into a virtual environment as provided by [venv](https://docs.python.org/3/library/venv.html). 3rd party alternatives to venv like [Poetry](https://python-poetry.org/) or [Conda](https://docs.conda.io/en/latest/) can also be used.

### Installation from PyPI (recommended)

Install `fancy_gym` via

```bash
pip install fancy_gym
```

We have a few optional dependencies. If you also want to install those use

```bash
# to install all optional dependencies
pip install 'fancy_gym[all]'

# or choose only those you want
pip install 'fancy_gym[dmc,box2d,mujoco-legacy,jax,testing]'
```

Pip can not automatically install up-to-date versions of metaworld, since they are not avaible on PyPI yet.
Install metaworld via

```bash
pip install metaworld@git+https://github.com/Farama-Foundation/Metaworld.git@d155d0051630bb365ea6a824e02c66c068947439#egg=metaworld
```

### Installation from master

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

We have a few optional dependencies. If you also want to install those use

```bash
# to install all optional dependencies
pip install -e '.[all]'

# or choose only those you want
pip install -e '.[dmc,box2d,mujoco-legacy,jax,testing]'
```

Metaworld has to be installed manually with

```bash
pip install metaworld@git+https://github.com/Farama-Foundation/Metaworld.git@d155d0051630bb365ea6a824e02c66c068947439#egg=metaworld
```

## How to use Fancy Gym

Documentation for `fancy_gym` is avaible at TODO:Link. Usage examples can be found here-TODO:Link.

### Step-Based Environments

Regular step based environments added by Fancy Gym are added into the `fancy/` namespace.

| &#x2757; Legacy versions of Fancy Gym used `fancy_gym.make(...)`. This is no longer supported and will raise an Exception on new versions. |
| ------------------------------------------------------------------------------------------------------------------------------------------ |

```python
import gymnasium as gym
import fancy_gym

env = gym.make('fancy/Reacher5d-v0')
# or env = gym.make('metaworld/reach-v2') # fancy_gym allows access to all metaworld ML1 tasks via the metaworld/ NS
# or env = gym.make('dm_control/ball_in_cup-catch-v0')
# or env = gym.make('Reacher-v2')
observation = env.reset(seed=1)

for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if i % 5 == 0:
        env.render()

    if terminated or truncated:
        observation, info = env.reset()
```

A list of all included environments is avaible here-TODO:Link.

### Black-box Environments

Existing MP tasks can be created the same way as above. The namespace of a MP-variant of an environment is given by `<original namespace>_<MP name>/`.
Just keep in mind, calling `step()` executes a full trajectory.

```python
import gymnasium as gym
import fancy_gym

env = gym.make('fancy_ProMP/Reacher5d-v0')
# or env = gym.make('metaworld_ProDMP/reach-v2')
# or env = gym.make('dm_control_DMP/ball_in_cup-catch-v0')
# or env = gym.make('gym_ProMP/Reacher-v2') # mp versions of envs added directly by gymnasium are in the gym_<MP-type> NS

# render() can be called once in the beginning with all necessary arguments.
# To turn it of again just call render() without any arguments.
env.render(mode='human')

# This returns the context information, not the full state observation
observation, info = env.reset(seed=1)

for i in range(5):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # terminated or truncated is always True as we are working on the episode level, hence we always reset()
    observation, info = env.reset()
```

A list of all included MP environments is avaible here-TODO:Link.

### How to create a new MP task

We refer to our Documentation for a complete description-TODO:Link.

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

As for how to write custom MP-Wrappers, please have a look at our Documentation-TODO:Link.
From this point on, you can access MP-version of your environments via

```python
env = gym.make('custom_ProDMP/cool_new_env-v0')

rewards = 0
observation, info = env.reset()

# number of samples/full trajectories (multiple environment steps)
for i in range(5):
    ac = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(ac)
    rewards += reward

    if terminated or truncated:
        print(rewards)
        rewards = 0
        observation, info = env.reset()
```

## Citing the Project

To cite this repository in publications:

```bibtex
@software{fancy_gym,
	title = {Fancy Gym},
	author = {Otto, Fabian and Celik, Onur and Roth, Dominik and Zhou, Hongyi},
	abstract = {Fancy Gym: Unifying interface for various RL benchmarks with support for Black Box approaches.},
	url = {https://github.com/ALRhub/fancy_gym},
	organization = {Autonomous Learning Robots Lab (ALR) at KIT},
}
```

## Icon Attribution

The icon is based on the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) icon as can be found [here](https://gymnasium.farama.org/_static/img/gymnasium_black.svg).
