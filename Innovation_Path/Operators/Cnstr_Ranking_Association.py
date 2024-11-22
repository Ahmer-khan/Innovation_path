import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from Innovation_Path.Operators.Ranking_and_Association import RankAndAssociation,Association_and_Crowding
from pymoo.core.survival import Survival, split_by_feasibility
from pymoo.core.population import Population
from pymoo.util.randomized_argsort import randomized_argsort
from Innovation_Path.core.PathProblem import Asf
from sklearn.preprocessing import MinMaxScaler as MMS
from random import random





class ConstrRankCrowdingAndAssociation(Survival):

    def __init__(self, z ,w ,cnst ,alpha ,zeta, nds=None, crowding_func="cd"):

        super().__init__(filter_infeasible=False)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.ranking = RankAndAssociation(cnst,alpha ,zeta,nds=nds)
        self.cnst = cnst
        self.alpha = alpha
        self.z = z
        self.w = w
        self.zeta = zeta

    def _do(self,
            problem,
            pop,
            *args,
            n_survive=None,
            **kwargs):

        if n_survive is None:
            n_survive = len(pop)

        n_survive = min(n_survive, len(pop))

        F = pop.get("F").astype(float, copy=False)
        X = pop.get("X").astype(float, copy=False)
        if problem.base.n_obj > 1:
            f_ori = pop.get("F_ori").astype(float, copy=False)
            norm = MMS()
            norm.fit(f_ori)       # osy

        # If the split should be done beforehand
        if problem.n_constr > 0:

            # Split by feasibility
            feas, infeas = split_by_feasibility(pop, sort_infeas_by_cv=True, sort_feas_by_obj=False, return_pop=False)

            # Obtain len of feasible
            n_feas = len(feas)

            # Assure there is at least_one survivor
            if n_feas == 0:
                if problem.base.n_obj > 1:
                    f_ori = norm.transform(f_ori)
                    f1    = Asf(f_ori ,norm.transform(self.z) ,self.w)             # z
                    obj_values = np.column_stack([f1 ,F[: ,1]])
                    pop.set("F", obj_values)

                survivors = Population()
            else:
                if problem.base.n_obj > 1:
                    norm.fit(f_ori[feas])
                    f_ori = norm.transform(f_ori)
                    f1    = Asf(f_ori ,norm.transform(self.z) ,self.w)     # z
                    obj_values = np.column_stack([f1 ,F[: ,1]])
                    pop.set("F", obj_values)
                survivors = self.ranking._do(problem, pop[feas], *args, n_survive=min(len(feas), n_survive), **kwargs)

            # Calculate how many individuals are still remaining to be filled up with infeasible ones
            n_remaining = n_survive - len(survivors)

            # If infeasible solutions need to be added
            if n_remaining > 0:

                # Constraints to new ranking
                G = pop[infeas].get("G")
                G = np.maximum(G, 0)
                H = pop[infeas].get("H")
                H = np.absolute(H)
                C = np.column_stack((G, H))
                # X = pop[infeas].get('X').astype(float, copy=False)
                F = pop.get('F').astype(float, copy=False)
                count_list = list(pop.get('counts'))

                # Fronts in infeasible population
                infeas_fronts = self.nds.do(C, n_stop_if_ranked=n_remaining)

                if len(survivors) > 0:

                    anch = np.where(pop.get('rank' )==0)[0]
                    max_rank = max(survivors.get('rank'))

                    anchorf = F[anch, :]
                    non_ind = anchorf[:, -1].argsort()
                    anch_count = [count_list[anch[i]] for i in non_ind]
                    count_list = anch_count[:]
                    curr_anch = 0
                    for i in range(len(count_list)):
                        if count_list[i] < self.zeta:
                            curr_anch = i
                            break

                    # Iterate over fronts
                    for k, front in enumerate(infeas_fronts):
                        I = []
                        Association_of_front, crowding_of_front = Association_and_Crowding(anch, front ,F,X, self.cnst
                                                                                           ,infeas)
                        counts = [-1 for i in range(len(front))]

                        if len(front) > n_remaining:
                            Associations = np.unique(Association_of_front)
                            if Associations.shape[0] == 1:
                                n_remove = len(survivors) + len(front) - n_survive
                                I = np.arange(len(front))
                                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')


                                I = I[:-n_remove]
                            else:
                                Associations = [i for i in range(len(count_list))]
                                j = len(Associations)
                                new = Associations[curr_anch:]
                                new.extend(Associations[:curr_anch][::-1])
                                Associations = new[:]
                                Associations = np.reshape(Associations, (j, 1))
                                a = 2 / ((1 + self.alpha) * (j))
                                weights = [a * (1 + (x * (self.alpha - 1)) / (j - 1)) for x in range(j)]
                                selected = [[] for n in range(len(infeas_fronts[0]))]
                                while len(I) < len(front):
                                    i = random()
                                    cum_prob = 0
                                    m = 0
                                    while j > 0:
                                        if i <= cum_prob + weights[m]:
                                            ind_temp = np.where(Association_of_front == Associations[m, 0])[0]
                                            index = [x for x in ind_temp if x not in selected[m]]
                                            if len(index) == 0:
                                                Associations = np.delete(Associations, m, axis=0)
                                                j = Associations.shape[0]
                                                if j > 1:
                                                    a = 2 / ((1 + self.alpha) * (j))
                                                    weights = [a * (1 + (x * (self.alpha - 1)) / (j - 1)) for x in
                                                               range(j)]
                                                    break
                                                else:
                                                    weights = [1]
                                                    break
                                            else:
                                                max_ind = np.argmax(crowding_of_front[index])
                                                I.append(index[max_ind])
                                                selected[m].append(index[max_ind])
                                                break
                                        else:
                                            cum_prob += weights[m]
                                            m += 1
                                if len(I) + len(survivors) >= n_survive:
                                    break
                        else:
                            I = np.arange(len(front))

                            # Save ranks
                        pop[infeas][front].set("cv_rank", k)
                        for j, i in enumerate(front):
                            pop[infeas[i]].set("rank", k + max_rank)
                            pop[infeas[i]].set('association', Association_of_front[j])
                            pop[infeas[i]].set("crowding", crowding_of_front[j])
                            pop[infeas[i]].set('counts', counts[j])

                        # extend the survivors by all or selected individuals
                        survivors = Population.merge(survivors, pop[infeas][front[I]])

                        if len(survivors) == n_survive:
                            break
                else:
                    for k, front in enumerate(infeas_fronts):
                        # Save ranks
                        pop[infeas][front].set("cv_rank", k)

                        # Current front sorted by CV
                        if len(survivors) + len(front) > n_survive:

                            # Obtain CV of front
                            CV = pop[infeas][front].get("CV").flatten()
                            I = randomized_argsort(CV, order='ascending', method='numpy')
                            I = I[:(n_survive - len(survivors))]

                        # Otherwise take the whole front unsorted
                        else:
                            I = np.arange(len(front))

                        # extend the survivors by all or selected individuals
                        survivors = Population.merge(survivors, pop[infeas][front[I]])
        else:
            if problem.base.n_obj > 1:
                f_ori = norm.transform(f_ori)
                f1 = Asf(f_ori, norm.transform(self.z), self.w)  # z
                obj_values = np.column_stack([f1, F[:, 1]])
                pop.set("F", obj_values)
            survivors = self.ranking._do(problem, pop, *args, n_survive=n_survive, **kwargs)

        return survivors