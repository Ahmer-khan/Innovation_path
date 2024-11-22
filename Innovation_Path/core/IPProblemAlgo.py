from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np
from Innovation_Path.Operators.Cnstr_Ranking_Association import ConstrRankCrowdingAndAssociation
from Innovation_Path.Operators.PathSampling import PathSampling
from Innovation_Path.Operators.PathSelection import PathSelection
from Innovation_Path.core.PathProblem import MyProblem,Asf

class IP_Problem_and_Algo(object):
    def __init__(self,pop_size,alpha,base,curr_sol,z,w,cnst,zeta):
        self.pop_size = pop_size
        self.alpha = alpha
        self.base = base
        self.curr_sol = curr_sol
        self.z = z
        self.w = w
        self.cnst = cnst
        self.zeta = zeta
    def create_problem(self):
        problem = MyProblem(self.curr_sol, self.base, 2, self.base.n_var, self.base.xl, self.base.xu, self.base.n_constr, self.base.n_ieq_constr,
                            self.base.n_eq_constr, self.z, self.w)
        return problem

    def create_algo(self):
        curr_F = Asf(self.base.evaluate(self.curr_sol)[0], self.z, self.w)
        curr_F = np.column_stack([curr_F[0], 0])
        algorithm = NSGA2(pop_size=self.pop_size,
                          sampling=PathSampling(self.curr_sol),
                          selection=PathSelection(self.alpha,self.zeta),
                          # crossover=SBX(eta=15, prob=0.9),
                          # mutation=PM(eta=20),
                          n_offsprings=self.pop_size,
                          survival=ConstrRankCrowdingAndAssociation(self.z, self.w,self.cnst, self.alpha,self.zeta))
                          # output=MultiObjectiveOutput())

        return algorithm