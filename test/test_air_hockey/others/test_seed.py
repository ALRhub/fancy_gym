import numpy as np
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper


if __name__ == "__main__":
    np.random.seed(90)
    env = AirHockeyChallengeWrapper('3dof-hit')
    env.base_env.seed(20)

    for i in range(3):
        obs = env.reset()
        print(obs)
