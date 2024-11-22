from pymoo.problems import get_problem
from pymoo.optimize import minimize
from core.IPProblemAlgo import IP_Problem_and_Algo as IP
import numpy as np
from utility.G1 import G1


alpha       = 0.05                   # the probability parameter gamma
pop_size    = 100                    # population size
base        = get_problem('G4')      # the base problem to be solved
curr_sol    = np.array([[100.3153175 ,  34.23543853,  40.47446649,  42.39405425, 37.93937681]]) # Current solution of the problem
z           = np.array([[0,0]])       # the aspiration point
w           = np.array([[0.8,0.2]])   # the prefernce vector for more than 1 objective
cnst        = G1(0.2)                 # Step Constraint Function intialized with delta

creator = IP(pop_size,alpha,base,curr_sol,z,w,cnst)
problem = creator.create_problem()
algorithm = creator.create_algo()

res = minimize(problem,
               algorithm,
               #termination,
               ('n_gen', 100),
               seed=1,
               verbose=True
              #callback = MyCallback()
              )