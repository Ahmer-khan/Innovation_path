class G1(object):
    def __init__(self,delta):
        self.delta = delta

    def calculate(self,x_a,f_a,x_p,f_p):
        diff = f_p[1] - f_a[1]
        CV   = self.delta - diff
        SCV  = max(0,CV)

        return CV,SCV,diff
