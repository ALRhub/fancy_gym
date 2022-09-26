# OpenAI Gym Wrappers

These are the Environment Wrappers for selected [OpenAI Gym](https://gym.openai.com/) environments to use
the Motion Primitive gym interface for them.

## MP Environments
These environments are wrapped-versions of their OpenAI-gym counterparts.

|Name| Description|Trajectory Horizon|Action Dimension|Context Dimension
|---|---|---|---|---|
|`ContinuousMountainCarProMP-v0`| A ProMP wrapped version of the ContinuousMountainCar-v0 environment. | 100 | 1
|`ReacherProMP-v2`| A ProMP wrapped version of the Reacher-v2 environment. | 50 | 2
|`FetchSlideDenseProMP-v1`| A ProMP wrapped version of the FetchSlideDense-v1 environment. | 50 | 4 
|`FetchReachDenseProMP-v1`| A ProMP wrapped version of the FetchReachDense-v1 environment. | 50 | 4
