from pymoo.core.survival import Survival #, split_by_feasibility
import numpy as np
from pymoo.util.randomized_argsort import randomized_argsort
from random import random
from Innovation_Path.Operators.PathSorting import PathSorting


class RankAndAssociation(Survival):

    def __init__(self, cnst, alpha,zeta, nds=None):

        super().__init__(filter_infeasible=True)
        self.nds = PathSorting()
        self.cnst  = cnst
        self.alpha = alpha
        self.zeta = zeta
    def _do(self,
            problem,
            pop,
            CV = None,
            *args,
            n_survive=None,
            **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)
        X = pop.get('X').astype(float, copy=False)
        count_list = list(pop.get('counts'))
        if count_list.count(None) == F.shape[0]:
            anchorf = []
            count_list = []
        else:
            rank = pop.get('rank')
            non_ind = np.where(rank == 0)[0]
            anchorf = F[non_ind, :]
            anch_count = [count_list[i] for i in list(non_ind)]
            count_list = anch_count[:]

        # the final indices of surviving individuals
        survivors = []
        # do the non-dominated sorting until splitting front
        fronts,cd_path,counts = self.nds.do(X,F,anchorf,count_list,self.cnst, n_stop_if_ranked=n_survive)
        curr_anch = 0
        for i in range(len(counts)):
            if counts[i] < self.zeta:
                curr_anch = i
                break


        for k, front in enumerate(fronts):

            if k == 0:
                I = np.arange(len(front))
                crowding_of_front = np.array(cd_path)
                Association_of_front = np.ones((len(cd_path))) * F.shape[0]

            else:
                Association_of_front, crowding_of_front = Association_and_Crowding(fronts[0], front,F,X, self.cnst)
                counts = [-1 for i in range(len(front))]

                if len(front) + len(survivors) <= n_survive:
                    I = np.arange(len(front))

                else:
                    Associations = np.unique(Association_of_front)
                    if Associations.shape[0] == 1:
                        n_remove = len(survivors) + len(front) - n_survive
                        I = np.arange(len(front))
                        I = randomized_argsort(crowding_of_front, order='descending', method='numpy')

                        I = I[:-n_remove]
                    else:
                        I = []
                        Associations = [i for i in range(len(fronts[0]))]
                        j = len(Associations)
                        new = Associations[curr_anch:]
                        new.extend(Associations[:curr_anch][::-1])
                        Associations = new[:]
                        Associations = np.reshape(Associations,(j,1))
                        a = 2/((1+self.alpha)*(j))
                        weights    = [a*(self.alpha + (((1 - self.alpha)*x)/(j-1))) for x in range(j)]
                        selected = [[] for n in range(len(fronts[0]))]
                        while len(I) < len(front):
                            i = random()
                            cum_prob = 0
                            m = 0
                            while j > 0:
                                if i <= cum_prob + weights[m]:
                                    ind_temp = np.where(Association_of_front == Associations[m,0])[0]
                                    index = [x for x in ind_temp if x not in selected]
                                    if len(index) == 0:
                                        Associations = np.delete(Associations,m,axis = 0)
                                        j = Associations.shape[0]
                                        if j > 1:
                                            a = 2/((1+self.alpha)*(j))
                                            weights = [a*(self.alpha + (((1 - self.alpha)*x)/(j-1))) for x in range(j)]
                                            break
                                        else:
                                            weights = [1]
                                            break
                                    else:
                                        max_ind = np.argmax(crowding_of_front[index])
                                        I.append(index[max_ind])
                                        selected.append(index[max_ind])
                                        break
                                else:
                                    cum_prob += weights[m]
                                    m += 1
                            if len(I) + len(survivors) >= n_survive:
                                break

            # save rank and crowding in the individual class
            for l, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[l])
                pop[i].set('association',Association_of_front[l])
                pop[i].set('counts', counts[l])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])
            if len(survivors) == n_survive:
                break
        return pop[survivors]


def Association_and_Crowding(ND,Front,F,X,cnst,in_ind = ''):
    SCV_mat = np.zeros((len(Front),len(ND)))
    G1_mat = np.zeros((len(Front), len(ND)))
    for i in range(len(Front)):
        if len(in_ind) == 0:
            point = F[Front[i],:]
            point_x = X[Front[i],:]
        else:
            point = F[in_ind[Front[i]],:]
            point_x = X[in_ind[Front[i]], :]
        for j in range(len(ND)):
            anchor = F[ND[j],:]
            anchor_x = X[ND[j],:]
            diff_f   = point[1] - anchor[1]
            _,_,diff,_,_   = cnst.calculate(anchor_x,anchor,point_x,point)
            SCV_mat[i,j] = diff
            G1_mat[i, j] = diff_f
    associations = np.zeros(len(Front))
    SCV          = np.zeros(len(Front))
    for i in range(SCV_mat.shape[0]):
        pos_ind      = np.where(G1_mat[i,:] >= 0)[0]
        if pos_ind.shape[0] > 0:
            if pos_ind.shape[0] == 1:
                SCV[i] = SCV_mat[i,:][pos_ind[0]]
                associations[i] = pos_ind[0]
            else:
                ind    = np.argmin(SCV_mat[i,:][pos_ind])
                SCV[i] = SCV_mat[i,pos_ind[ind]]
                associations[i] = pos_ind[ind]
        else:
            ind     = np.argmax(SCV_mat[i,:])
            SCV[i]  = SCV_mat[i,ind]
            associations[i] = ind
    return associations,SCV