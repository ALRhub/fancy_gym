class AlrReward:
    """
    A base class for non-Markovian reward functions which may need trajectory information to calculate an episodic
    reward. Call the methods in reset() and step() of the environment.
    """

    # methods to override:
    # ----------------------------
    def reset(self, *args, **kwargs):
        """
        Reset the reward function, empty state buffers before an episode, set contexts that influence reward, etc.
        """
        raise NotImplementedError

    def compute_reward(self, *args, **kwargs):
        """

        Returns: Useful things to return are reward values, success flags or crash flags

        """
        raise NotImplementedError
