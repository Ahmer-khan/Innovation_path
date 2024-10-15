from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np
from Operators.Cnstr_Ranking_Association import ConstrRankCrowdingAndAssociation
from Operators.PathSampling import PathSampling
from Operators.PathSelection import PathSelection
from core.PathProblem import MyProblem,Asf

class IP_Problem_and_Algo(object):
    def __init__(self,pop_size,alpha,base,curr_sol,z,w,cnst):
        self.pop_size = pop_size
        self.alpha = alpha
        self.base = base
        self.curr_sol = curr_sol
        self.z = z
        self.w = w
        self.cnst = cnst
    def create_problem(self):
        problem = MyProblem(self.curr_sol, self.base, 2, self.base.n_var, self.base.xl, self.base.xu, self.base.n_constr, self.base.n_ieq_constr,
                            self.base.n_eq_constr, self.z, self.w)
        return problem

    def create_algo(self):
        curr_F = Asf(self.base.evaluate(self.curr_sol)[0], self.z, self.w)
        curr_F = np.column_stack([curr_F[0], 0])
        algorithm = NSGA2(pop_size=self.pop_size,
                          sampling=PathSampling(self.curr_sol),
                          selection=PathSelection(self.alpha),
                          # crossover=SBX(eta=15, prob=0.9),
                          # mutation=PM(eta=20),
                          n_offsprings=self.pop_size,
                          survival=ConstrRankCrowdingAndAssociation(self.z, self.w,self.cnst, self.curr_sol, self.alpha, curr_F))
                          # output=MultiObjectiveOutput())

        return algorithm