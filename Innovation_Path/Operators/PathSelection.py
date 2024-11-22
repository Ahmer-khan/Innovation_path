import numpy as np
from random import sample,random
from pymoo.core.selection import Selection



class PathSelection(Selection):
    """
      The Path selection is used to simulated a tournament between individuals for each association in a round robin
      fashion to generate parents. The pressure balances greedy the genetic algorithm will be.
    """

    def __init__(self, alpha,zeta, func_comp=None, pressure=2, **kwargs):
        """

        Parameters
        ----------
        func_comp: func
            The function to compare two individuals. It has the shape: comp(pop, indices) and returns the winner.
            If the function is None it is assumed the population is sorted by a criterium and only indices are compared.

        pressure: int
            The selection pressure to bie applied. Default it is a binary tournament.
        """

        super().__init__(**kwargs)

        # selection pressure to be applied
        self.pressure = pressure
        self.func_comp = func_comp
        self.alpha  = alpha
        self.zeta = zeta
        # if self.func_comp is None:
        # raise Exception("Please provide the comparing function for the tournament selection!")

    def _do(self, _, pop, n_select ,n_parents=2, **kwargs):

        S = []
        F = pop.get('F').astype(float, copy=False)
        rank = pop.get('rank')
        crowd = pop.get('crowding')
        asso  = pop.get('association')
        counts = list(pop.get('counts'))
        non_ind = np.where(rank == 0)[0]
        non_values = F[non_ind, :]
        non_ind_arr = non_values[:, -1].argsort()
        anch_count = [counts[non_ind[i]] for i in non_ind_arr]
        counts = anch_count[:]
        curr_anch = 0
        for i in range(len(counts)):
            if counts[i] < self.zeta:
                curr_anch = i
                break
        Associations = [i for i in range(non_ind.shape[0])]
        k = len(Associations)
        if k > 1:
            a = 2/ ((1 + self.alpha) * (k))
            weights = [a * (self.alpha + (((1 - self.alpha) * x) / (k - 1))) for x in range(k)]
        else:
            weights = [1]
        count = 0
        while count < n_select:
            if curr_anch == len(counts) - 1:
                Parents = [pop[non_ind[Associations[-1]]]]
                pot_ind = np.arange(len(rank))
            else:
                i = random()
                cum_prob = 0
                j = 0
                Parents = []
                while len(Parents) < 1:
                    if i <= cum_prob + weights[j]:
                        Parents = [pop[non_ind[Associations[j]]]]
                        pot_ind = np.where(asso == Associations[j])[0]
                        if len(pot_ind) == 0:
                            Associations.pop(j)
                            k = len(Associations)
                            if k > 1:
                                a = 2 / ((1 + self.alpha) * (k))
                                weights = [a * (1 + (x * (self.alpha - 1)) / (k - 1)) for x in range(k)]
                            else:
                                weights = [1]
                            i = random()
                            cum_prob = 0
                            j = 0
                            Parents = []
                    else:
                        cum_prob += weights[j]
                        j += 1

            if self.pressure > len(pot_ind):
                press = len(pot_ind)
            else:
                press = self.pressure
            sec = sample(list(pot_ind), press)
            rank_min = np.min(rank[sec])
            min_count = np.where(rank[sec] == rank_min)[0]
            if len(min_count) > 1:
                crowd_max = np.argmax(crowd[sec])
                Parents.append(pop[sec[crowd_max]])
            else:
                Parents.append(pop[sec[min_count[0]]])

            S.append(np.array(Parents))
            count += 1

        S = np.array(S)

        return np.reshape(S, (n_select, n_parents))