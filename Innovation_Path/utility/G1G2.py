import numpy as np


class G1G2(object):
    def __init__(self, delta):
        self.delta = delta

    def calculate(self, x_a, f_a, x_p, f_p):
        CV_mat = np.zeros((1,2))
        diff1 = np.linalg.norm(x_a - x_p)
        diff2 = f_p[1] - f_a[1]
        diff  = diff1/self.delta[1] + diff2/self.delta[0]
        CV_mat[0,0] = self.delta[1] - diff1
        CV_mat[0,1] = self.delta[0] - diff2
        CV = CV_mat[0,0]/self.delta[1] + CV_mat[0,1]/self.delta[0]
        SCV = max(0, CV_mat[0,0]) + max(0,CV_mat[0,1])

        return CV, SCV,diff