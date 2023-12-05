# DeepMind Control (DMC)

These are the Environment Wrappers for selected
[DeepMind Control](https://deepmind.com/research/publications/2020/dm-control-Software-and-Tasks-for-Continuous-Control)
environments in order to use our Motion Primitive gym interface with them.

## MP Environments

[//]: <> (These environments are wrapped-versions of their Deep Mind Control Suite &#40;DMC&#41; counterparts. Given most task can be)
[//]: <> (solved in shorter horizon lengths than the original 1000 steps, we often shorten the episodes for those task.)

| Name                                     | Description                                                                    | Trajectory Horizon | Action Dimension | Context Dimension |
| ---------------------------------------- | ------------------------------------------------------------------------------ | ------------------ | ---------------- | ----------------- |
| `dm_control_ProDMP/ball_in_cup-catch-v0` | A ProMP wrapped version of the "catch" task for the "ball_in_cup" environment. | 1000               | 10               | 2                 |
| `dm_control_DMP/ball_in_cup-catch-v0`    | A DMP wrapped version of the "catch" task for the "ball_in_cup" environment.   | 1000               | 10               | 2                 |
| `dm_control_ProDMP/reacher-easy-v0`      | A ProMP wrapped version of the "easy" task for the "reacher" environment.      | 1000               | 10               | 4                 |
| `dm_control_DMP/reacher-easy-v0`         | A DMP wrapped version of the "easy" task for the "reacher" environment.        | 1000               | 10               | 4                 |
| `dm_control_ProDMP/reacher-hard-v0`      | A ProMP wrapped version of the "hard" task for the "reacher" environment.      | 1000               | 10               | 4                 |
| `dm_control_DMP/reacher-hard-v0`         | A DMP wrapped version of the "hard" task for the "reacher" environment.        | 1000               | 10               | 4                 |
