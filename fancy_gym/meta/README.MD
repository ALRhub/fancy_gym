# MetaWorld Wrappers

These are the Environment Wrappers for selected [Metaworld](https://meta-world.github.io/) environments in order to use our Movement Primitive gym interface with them. 
All Metaworld environments have a 39 dimensional observation space with the same structure. The tasks differ only in the objective and the initial observations that are randomized. 
Unused observations are zeroed out. E.g. for `Button-Press-v2` the observation mask looks the following:
```python
    return np.hstack([
        # Current observation
        [False] * 3,  # end-effector position
        [False] * 1,  # normalized gripper open distance
        [True] * 3,  # main object position
        [False] * 4,  # main object quaternion
        [False] * 3,  # secondary object position
        [False] * 4,  # secondary object quaternion
        # Previous observation
        [False] * 3,  # previous end-effector position
        [False] * 1,  # previous normalized gripper open distance
        [False] * 3,  # previous main object position
        [False] * 4,  # previous main object quaternion
        [False] * 3,  # previous second object position
        [False] * 4,  # previous second object quaternion
        # Goal
        [True] * 3,  # goal position
    ])
```
For other tasks only the boolean values have to be adjusted accordingly.