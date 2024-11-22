import numpy as np
class G1(object):
    def __init__(self,delta):
        self.delta = delta

    def calculate(self,x_a,f_a,x_p,f_p):
        diff = f_p[1] - f_a[1]
        CV   = self.delta - diff
        SCV  = max(0,CV)
        del1 = -(f_a[0] - f_p[0])/f_a[0]
        point_mat = np.array([del1,diff/self.delta])
        step_mat = np.array([-1,1])

        return CV,SCV,diff,point_mat,step_mat
