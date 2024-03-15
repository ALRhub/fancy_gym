<h1 align="center">
  <br>
  <img src='https://raw.githubusercontent.com/ALRhub/fancy_gym/master/icon.svg' width="250px">
  <br><br>
  <b>Fancy Gym</b>
  <br><br>
</h1>

Built upon the foundation of [Gymnasium](https://gymnasium.farama.org) (a maintained fork of OpenAI’s renowned Gym library) `fancy_gym` offers a comprehensive collection of reinforcement learning environments.

**Key Features**:

- **New Challenging Environments**: `fancy_gym` includes several new environments ([Panda Box Pushing](https://alrhub.github.io/fancy_gym/envs/fancy/mujoco.html#box-pushing), [Table Tennis](https://alrhub.github.io/fancy_gym/envs/fancy/mujoco.html#table-tennis), [etc.](https://alrhub.github.io/fancy_gym/envs/fancy/index.html)) that present a higher degree of difficulty, pushing the boundaries of reinforcement learning research.
- **Support for Movement Primitives**: `fancy_gym` supports a range of movement primitives (MPs), including Dynamic Movement Primitives (DMPs), Probabilistic Movement Primitives (ProMP), and Probabilistic Dynamic Movement Primitives (ProDMP).
- **Upgrade to Movement Primitives**: With our framework, it’s straightforward to transform standard Gymnasium environments into environments that support movement primitives.
- **Benchmark Suite Compatibility**: `fancy_gym` makes it easy to access renowned benchmark suites such as [DeepMind Control](https://alrhub.github.io/fancy_gym/envs/dmc.html)
  and [Metaworld](https://alrhub.github.io/fancy_gym/envs/meta.html), whether you want to use them in the regular step-based setting or using MPs.
- **Contribute Your Own Environments**: If you’re inspired to create custom gym environments, both step-based and with movement primitives, this [guide](https://alrhub.github.io/fancy_gym/guide/upgrading_envs.html) will assist you. We encourage and highly appreciate submissions via PRs to integrate these environments into `fancy_gym`.

## Quickstart Guide

| &#x26A0; We recommend installing `fancy_gym` into a virtual environment as provided by [venv](https://docs.python.org/3/library/venv.html), [Poetry](https://python-poetry.org/) or [Conda](https://docs.conda.io/en/latest/). |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |

Install via pip [or use an alternative installation method](https://alrhub.github.io/fancy_gym/guide/installation.html)

```bash
    pip install 'fancy_gym[all]'
```

Try out one of our step-based environments [or explore our other envs](https://alrhub.github.io/fancy_gym/envs/fancy/index.html)

```python
   import gymnasium as gym
   import fancy_gym
   import time

   env = gym.make('fancy/BoxPushingDense-v0', render_mode='human')
   observation = env.reset()
   env.render()

   for i in range(1000):
      action = env.action_space.sample() # Randomly sample an action
      observation, reward, terminated, truncated, info = env.step(action)
      time.sleep(1/env.metadata['render_fps'])

      if terminated or truncated:
            observation, info = env.reset()
```

Explore the MP-based variant [or learn more about Movement Primitives (MPs)](https://alrhub.github.io/fancy_gym/guide/episodic_rl.html)

```python
   import gymnasium as gym
   import fancy_gym

   env = gym.make('fancy_ProMP/BoxPushingDense-v0', render_mode='human')
   env.reset()
   env.render()

   for i in range(10):
      action = env.action_space.sample() # Randomly sample MP parameters
      observation, reward, terminated, truncated, info = env.step(action) # Will execute full trajectory, based on MP
      observation = env.reset()
```

## Documentation

Documentation for `fancy_gym` can be found [here](https://alrhub.github.io/fancy_gym/); Usage Examples can be found [here](https://alrhub.github.io/fancy_gym/examples/general.html).

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
