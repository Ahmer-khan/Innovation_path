import numpy as np


class G2(object):
    def __init__(self, delta):
        self.delta = delta

    def calculate(self, x_a, f_a, x_p, f_p):
        diff = np.linalg.norm(x_a - x_p)
        CV = self.delta - diff
        SCV = max(0, CV)

        return CV, SCV,diff