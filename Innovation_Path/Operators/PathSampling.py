from pymoo.core.sampling import Sampling
import numpy as np


class PathSampling(Sampling):

    def __init__(self, curr) -> None:
        super().__init__()
        self.curr = curr

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.random((n_samples, problem.n_var))

        if problem.has_bounds():
            xl, xu = problem.bounds()
            assert np.all(xu >= xl)
            X = xl + (xu - xl) * X
        X[0, :] = self.curr

        return X