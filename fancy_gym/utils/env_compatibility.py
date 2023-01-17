import gymnasium as gym


class EnvCompatibility(gym.wrappers.EnvCompatibility):
    def __getattr__(self, item):
        """Propagate only non-existent properties to wrapped env."""
        if item.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(item))
        if item in self.__dict__:
            return getattr(self, item)
        return getattr(self.env, item)
