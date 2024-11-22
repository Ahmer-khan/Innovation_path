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

    def do(self, X, F ,anch_val,counts,cnst, return_rank=False, only_non_dominated_front=False, n_stop_if_ranked=None, **kwargs):
        if n_stop_if_ranked is None:
            n_stop_if_ranked = int(1e8)
        first_ind = F[:, 0].argsort()  # [::-1]
        temp = F[[ind for ind in first_ind]]
        temp_X = X[[ind for ind in first_ind]]
        non_ind = temp[:, -1].argsort()
        if np.min(temp[:,0]) < 0 and np.max(temp[:,0]) < 0:
            val_temp   = temp[:,0] - np.min(temp[:,0])
            temp = np.column_stack([val_temp,temp[:,1]])
        fronts_new = []
        cd_path    = [0]
        flag       = False
        A          = 0
        dom_mat = np.zeros((F.shape[0], F.shape[0]))
        nex_can = 0
        if len(counts) == 0:
            counts.append(0)
            flag = True

        for i in range(dom_mat.shape[0]):
            if i == nex_can:
                step_check = True
                mini = np.inf
                dom_mat[i, :] = 0
                A += 1
            else:
                step_check = False

            for j in range(i + 1, dom_mat.shape[0]):

                if temp[non_ind[j], 1] >= temp[non_ind[i], 1] and temp[non_ind[i], 0] < temp[non_ind[j], 0]:
                    dom_mat[j, i] = 1  # dom_sym
                elif temp[non_ind[j], 1] > temp[non_ind[i], 1] and temp[non_ind[i], 0] <= temp[non_ind[j], 0]:
                    dom_mat[j, i] = 1  # dom_sym

                elif step_check:
                    SCV,diff,_,point_mat,step_mat = cnst.calculate(temp_X[non_ind[i],:],temp[non_ind[i], :]
                                                                   ,temp_X[non_ind[j],:],temp[non_ind[j], :])

                    if diff > 0:
                        dom_mat[j, i] = 1
                    else:
                        vio = perpendicular_distance(point_mat, step_mat)
                        # print(vio,cross)
                        if vio < mini:
                            if mini != np.inf:
                                for k in range(nex_can, j):
                                    dom_mat[k, i] = 1
                            mini = vio
                            nex_can = j
                            try:
                                cd_path[A] = SCV
                            except IndexError:
                                cd_path.append(SCV)
        while True:
            front = np.where(np.sum(dom_mat, axis=1, keepdims=True) == 0)[0]
            if len(front) == 0:
                break
            indices = []
            if len(fronts_new) == 0:
                for i in range(front.shape[0]):
                    index = first_ind[non_ind[front[i]]]
                    indices.append(index)
                    if i > 0:
                        if i < len(anch_val) and flag == False:
                            change = anch_val[i, 1] - F[index,1]
                            if change <= 1e-4 and change >= 0:
                                if anch_val[i, 0] - F[index,0] <= 1e-4 and anch_val[i+1, 0] - F[index,0] >= 0:
                                    counts[i-1] += 1
                                else:
                                    counts[i-1] = 0
                                    flag = True
                                # counts[index-1] += 1
                            else:
                                counts[i-1] = 0
                                flag = True
                        elif flag == True and i+1 < len(anch_val):
                            counts[i] = 0
                        else:
                            counts.append(0)
                fronts_new.append(np.array(indices))
            else:
                fronts_new.append(np.array([first_ind[non_ind[i]] for i in front]))
            for ele in front:
                dom_mat[:, ele] = 0
                dom_mat[ele, :] = 2

        if len(counts) < len(fronts_new[0]):
            for i in range(len(fronts_new[0]) - len(counts)):
                counts.append(0)
        if len(counts) > len(fronts_new[0]):
            counts = counts[:len(fronts_new[0])]
            counts[-1] = 0


        if only_non_dominated_front:
            return fronts_new[0] ,cd_path

        if return_rank:
            rank = rank_from_fronts(fronts_new, F.shape[0])
            return fronts_new, rank ,cd_path


        return fronts_new ,cd_path,counts


def rank_from_fronts(fronts, n):
    # create the rank array and set values
    rank = np.full(n, 1e16, dtype=int)
    for i, front in enumerate(fronts):
        rank[front] = i

    return rank


def perpendicular_distance(point, direction):
    # Convert lists to numpy arrays
    point = np.array(point)
    direction = np.array(direction)

    # Step 1: Normalize the direction vector
    norm_direction = direction / np.linalg.norm(direction)

    # Step 2: Project the point onto the normalized direction vector
    projection_length = np.dot(point, norm_direction)
    projection = projection_length * norm_direction

    # Step 3: Compute the perpendicular distance
    perpendicular_vector = point - projection
    distance = np.linalg.norm(perpendicular_vector)

    return distance