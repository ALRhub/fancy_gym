# Gymnasium

<div class='center'>
    <img src="../_static/imgs/env_gifs/Lunar_Lander.gif" style="margin: 5%; width: 60%;">
    <p>Lunar Lander Task (LunarLander-v2)</p>
</div>
<br>

We provide MP versions for selected [Farama Gymnasium](https://gymnasium.farama.org/) (previously OpenAI Gym) environments.

## Step-Based Environments

We refer to the [Gymnasium docs](https://gymnasium.farama.org/content/basic_usage/) for an overview of step-based environments provided by them.

## MP Environments

These environments are wrapped-versions of their Gymnasium counterparts.

| Name                                 | Description                                                          | Trajectory Horizon | Action Dimension |
| ------------------------------------ | -------------------------------------------------------------------- | ------------------ | ---------------- |
| `gym_ProMP/ContinuousMountainCar-v0` | A ProMP wrapped version of the ContinuousMountainCar-v0 environment. | 100                | 1                |
| `gym_ProMP/Reacher-v2`               | A ProMP wrapped version of the Reacher-v2 environment.               | 50                 | 2                |
| `gym_ProMP/FetchSlideDense-v1`       | A ProMP wrapped version of the FetchSlideDense-v1 environment.       | 50                 | 4                |
| `gym_ProMP/FetchReachDense-v1`       | A ProMP wrapped version of the FetchReachDense-v1 environment.       | 50                 | 4                |
