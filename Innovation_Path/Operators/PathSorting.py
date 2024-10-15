from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np


class PathSorting:
    def __init__(self, epsilon=None, method="fast_non_dominated_sort") -> None:
        super().__init__()
        self.epsilon = epsilon
        self.method = method
        '''self.eu_tol = eu_tol
        self.x0 = curr
        self.xF = currf'''

    def do(self, X, F ,cnst ,x0 ,xF, return_rank=False, only_non_dominated_front=False, n_stop_if_ranked=None, **kwargs):
        if n_stop_if_ranked is None:
            n_stop_if_ranked = int(1e8)
        fronts_new = []
        n_ranked   = 0
        cd_path    = [np.inf]
        prev       = x0
        prevf      = xF[0,:]
        fronts     = NonDominatedSorting(self.epsilon ,self.method).do(F)
        front      = fronts[0]
        non_values = F[front ,:]
        non_ind    = non_values[: ,-1].argsort()
        points     = [front[ind] for ind in non_ind[1:]]
        Anchors    = [front[non_ind[0]]]
        indices    = []

        for i in range(len(points)):
            val_I = X[points[i] ,:].T
            I     = F[points[i] ,:]
            SCV,diff,_ = cnst.calculate(prev,prevf,val_I,I)


            if  diff == 0:
                Anchors.append(points[i])
                prev = val_I
                prevf = I
                cd_path.append(SCV)
                n_ranked += 1

            else:
                indices.append(points[i])


        fronts_new.append(np.array(Anchors))
        if n_ranked < n_stop_if_ranked and len(indices) > 0:
            fronts_new.append(np.array(indices))
            n_ranked += len(indices)

        if n_ranked < n_stop_if_ranked:
            for front in fronts[1:]:
                fronts_new.append(front)
                n_ranked += front.shape[0]
                if n_ranked >= n_stop_if_ranked:
                    break


        if only_non_dominated_front:
            return fronts_new[0] ,cd_path

        if return_rank:
            rank = rank_from_fronts(fronts_new, F.shape[0])
            return fronts_new, rank ,cd_path


        return fronts_new ,cd_path


def rank_from_fronts(fronts, n):
    # create the rank array and set values
    rank = np.full(n, 1e16, dtype=int)
    for i, front in enumerate(fronts):
        rank[front] = i

    return rank