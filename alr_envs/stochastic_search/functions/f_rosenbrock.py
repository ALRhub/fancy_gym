import numpy as np

from alr_envs.stochastic_search.functions.f_base import BaseObjective


class Rosenbrock(BaseObjective):
    def __init__(self, dim, int_opt=(-3., 3.)):
        super(Rosenbrock, self).__init__(dim, int_opt=int_opt)
        self.c = np.maximum(1, np.sqrt(self.dim) / 8)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        z = self.c * (x - self.x_opt) + 1
        z_end = z[:, 1:]
        z_begin = z[:, :-1]

        a = z_begin ** 2 - z_end
        b = z_begin - 1

        return np.sum(100 * a ** 2 + b ** 2, axis=1) + self.f_opt


class RosenbrockRotated(BaseObjective):
    def __init__(self, dim, int_opt=(-3., 3.)):
        super(RosenbrockRotated, self).__init__(dim, int_opt=int_opt)
        self.c = np.maximum(1, np.sqrt(self.dim) / 8)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        z = (self.c * self.r @ x.T + 1 / 2).T
        a = z[:, :-1] ** 2 - z[:, 1:]
        b = z[:, :-1] - 1

        return np.sum(100 * a ** 2 + b ** 2, axis=1) + self.f_opt


class RosenbrockRaw(BaseObjective):
    def __init__(self, dim, int_opt=(-3., 3.)):
        super(RosenbrockRaw, self).__init__(dim, int_opt=int_opt)
        self.x_opt = np.ones((1, dim))
        self.f_opt = 0

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        a = x[:, :-1] ** 2 - x[:, 1:]
        b = x[:, :-1] - 1

        out = np.sum(100 * a ** 2 + b ** 2, axis=1)

        return out
