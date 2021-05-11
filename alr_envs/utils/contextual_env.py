from abc import abstractmethod
from typing import Union, Tuple

import numpy as np
from gym import Env

class ContextualEnv(Env):
    """A contextual environment. It functions just as any regular OpenAI Gym environment but it includes
    a context space and a method to set the context of the environment.
    """
    context_space = None # :gym.spaces

    @abstractmethod
    def set_context(self, context: Union[Tuple, float, np.ndarray, int]):
        """Set the environments context. This externalizes the context selection and allows
        to include learning of its distribution

        :param context: Context to be applied onto the environment
        """
        raise NotImplementedError