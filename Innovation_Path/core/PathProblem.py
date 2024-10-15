import numpy as np
from pymoo.core.problem import Problem
from sklearn.preprocessing import MinMaxScaler as MMS


class MyProblem(Problem):

    def __init__(self ,curr ,base ,n_obj ,n_var ,xl ,xu, n_constr ,n_ieq_constr ,n_eq_constr ,z ,w):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, n_constr=n_constr, n_ieq_constr = n_ieq_constr, n_eq_constr = n_eq_constr, vtype=float)
        self.curr = curr
        self.z    = z
        self.w    = w
        self.xl   = xl
        self.xu   = xu
        self.base = base
        self.n_ieq_constr = n_ieq_constr
        self.n_eq_constr  = n_eq_constr
    def _evaluate(self, x, out, *args, **kwargs):
        if self.n_ieq_constr > 0 and self.n_eq_constr > 0:
            f ,g ,h = self.base.evaluate(x)
            out['G'] = g
            out['H'] = h
        elif self.n_ieq_constr > 0:
            f ,g = self.base.evaluate(x)
            out['G'] = g
        elif self.n_eq_constr > 0:
            f ,h = self.base.evaluate(x)
            out['H'] = h
        else:
            f = self.base.evaluate(x)

        scalar = MMS()
        scalar.fit([self.xl,self.xu])
        if self.base.n_obj > 1:
            f1 = Asf(f ,self.z ,self.w)
            out['F_ori'] = f
        else:
            f1 = f
        f3 = diff(self.curr ,x,scalar)
        out['F'] = np.column_stack([f1, f3])

def diff(curr ,X,scalar):
    f_val = []
    X_new = scalar.transform(X)
    curr  = scalar.transform(curr)
    for i in range(X.shape[0]):
        f_val.append(np.linalg.norm((curr[0] - X_new[i ,:].T)))
    return f_val

def Asf(X ,asp ,weight):
    f = np.max(((X - asp ) /weight) ,axis = 1)
    return f