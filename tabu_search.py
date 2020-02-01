import numpy as np
from collections import deque

n_fvals = 0
best_fval_history = []


def update_stm(stm, x, stm_size):
    stm.append(tuple(x))
    if len(stm) > stm_size:
        stm.popleft()


def update_mtm(mtm, x, fval, mtm_size):    
    x = tuple(x)
    if x not in mtm:
        if len(mtm) < mtm_size:
            mtm[x] = fval
        else:
            # replace the worst solution in MTM with x
            worst_x = max(mtm, key=mtm.get)
            if fval < mtm[worst_x]:
                mtm.pop(worst_x)
                mtm[x] = fval


def update_ltm(ltm, x, bounds, n_sectors):
    if np.any((x < bounds[0]) | (x > bounds[1])):
        # do not update ltm if out of bound
        return

    # subtract a small epsilon if any component = upper bound (for floor division)
    x[x == bounds[1]] = bounds[1] - 1e-10

    sector_length = (bounds[1] - bounds[0]) / n_sectors
    # calculate sector numbers of x
    sector = tuple((x - bounds[0]) // sector_length)
    if sector not in ltm:
        ltm[sector] = {tuple(x)}
    else:
        ltm[sector].add(tuple(x))


def local_search(x0, obj_func, bounds, step_size, stm):
    fval0 = obj_func(x0)
    best_x = None
    best_fval = fval0

    for i in range(x0.shape[0]):
        x = x0.copy()
        x[i] += step_size
        # check that location is not in stm and is valid
        if tuple(x) not in stm and np.all((x >= bounds[0]) & (x <= bounds[1])):
            tmp_fval = obj_func(x)
            if best_x is None or obj_func(x) < best_fval:
                best_x = x
                best_fval = tmp_fval

        # do the same thing for x_i - step_i
        x = x0.copy()
        x[i] -= step_size
        if tuple(x) not in stm and np.all((x >= bounds[0]) & (x <= bounds[1])):
            tmp_fval = obj_func(x)
            if best_x is None or obj_func(x) < best_fval:
                best_x = x
                best_fval = tmp_fval

    if best_x is None:
        # only happens when all solutions are rejected
        return x0, fval0

    # pattern move
    tmp_x = best_x + best_x - x0
    if tuple(tmp_x) not in stm and np.all((tmp_x >= bounds[0]) & (tmp_x <= bounds[1])):
        tmp_fval = obj_func(tmp_x)
        if best_fval < fval0 and tmp_fval < best_fval:
            best_x = tmp_x
            best_fval = tmp_fval

    return best_x, best_fval


def intensify(mtm, n):
    # return mean location in MTM
    X = np.array(list(mtm.keys()))
    return X.mean(axis=0)


def diversify(ltm, n, n_sectors, bounds, explored_thresh):
    if len(ltm) == n_sectors ** n and \
            all([len(locs) > explored_thresh for locs in ltm.values()]):
        # all areas have been reasonably searched; pick the one with the smallest count
        sector = np.array(min(ltm, key=lambda k: len(ltm[k])))
    
    else:
        # randomly select an unexplored sector
        sector = np.random.randint(n_sectors, size=n)
        while tuple(sector) in ltm and len(ltm[tuple(sector)]) > explored_thresh:
            # if it is explored already, select another sector
            sector = np.random.randint(n_sectors, size=n)

    # sample uniformly in sector
    sector_length = (bounds[1] - bounds[0]) / n_sectors
    x = bounds[0] + sector_length * sector + np.random.uniform(0, sector_length, size=n)

    return x


def tabu_search(x, obj_func, n, bounds=(-2, 2), step_size=0.1, n_sectors=10,
                si_thresh=10, sd_thresh=15, ssr_thresh=25,
                stm_size=3, mtm_size=4, ssr_factor=0.5, explored_thresh=10,
                epsilon=1e-9, max_n_fvals=1e4):
    try:
        reset()

        stm = deque()       # queue data structure
        mtm = {}    # <tuple> x: <float> fval
        ltm = {}    # <tuple> sector: <set> visited_locations

        best_x = x
        best_fval = obj_func(x)

        # save intermediate results for analysis
        history = {'x': [x], 'fval':[best_fval], 'si': [], 'sd': [], 'ssr': [],
                   'convergence': -1}

        update_stm(stm, best_x, stm_size)
        update_mtm(mtm, best_x, best_fval, mtm_size)
        update_ltm(ltm, best_x, bounds, n_sectors)

        counter = 0
        while n_fvals <= max_n_fvals:
            if history['convergence'] == -1 and step_size < epsilon:
                # convergence criteria met
                history['convergence'] = n_fvals

            x, fval = local_search(x, obj_func, bounds, step_size, stm)
            
            history['x'].append(x)
            history['fval'].append(fval)

            update_stm(stm, x, stm_size)
            update_mtm(mtm, x, fval, mtm_size)
            update_ltm(ltm, x, bounds, n_sectors)

            if fval < best_fval:     # new best solution found
                best_x, best_fval = x, fval
                counter = 0     # reset counter
                continue    # ignore the remaining code in this iteration

            if counter == si_thresh:
                history['si'].append(n_fvals)

                x = intensify(mtm, n)
                fval = obj_func(x)

                history['x'].append(x)
                history['fval'][-1] = fval

                update_stm(stm, x, stm_size)
                update_mtm(mtm, x, fval, mtm_size)
                update_ltm(ltm, x, bounds, n_sectors)

                if fval < best_fval:
                    best_x, best_fval = x, fval
                    counter = 0     # reset counter
                    continue

            elif counter == sd_thresh:
                history['sd'].append(n_fvals)

                x = diversify(ltm, n, n_sectors, bounds, explored_thresh=explored_thresh)
                fval = obj_func(x)

                history['x'].append(x)
                history['fval'][-1] = fval

                update_stm(stm, x, stm_size)
                update_mtm(mtm, x, fval, mtm_size)
                update_ltm(ltm, x, bounds, n_sectors)

                if fval < best_fval:
                    best_x, best_fval = x, fval
                    counter = 0     # reset counter
                    continue

            elif counter == ssr_thresh:
                history['ssr'].append(n_fvals)

                step_size *= ssr_factor     # reduce step size
                x = best_x      # move x to best location

                history['x'].append(x)
                history['fval'].append(fval)

                update_stm(stm, x, stm_size)
                # no need to update MTM and LTM since x is already in them

                counter = 0     # reset counter
                continue

            counter += 1
    
    except MaxFuncEvaluationsExceeded:
        history['best_fval'] = best_fval_history.copy()
        return best_x, best_fval, history

    history['best_fval'] = best_fval_history.copy()
    return best_x, best_fval, history


def reset():
    global n_fvals, start_time
    n_fvals = 0
    best_fval_history.clear()


def shubert(x):
    a = np.arange(1, 6)
    A = np.tile(a, (x.shape[0], 1))
    fval = np.sum( np.dot( np.sin((A + 1) * np.expand_dims(x, -1) + A), a ) )
    return fval


def shubert_with_count(x, max_n_fvals):
    global n_fvals
    if n_fvals >= max_n_fvals:
        # raise exception to be caught by try block
        raise MaxFuncEvaluationsExceeded()
    
    fval = shubert(x)

    n_fvals += 1
    if not best_fval_history or fval < best_fval_history[-1]:
        best_fval_history.append(fval)
    else:
        best_fval_history.append(best_fval_history[-1])
    
    return fval


class MaxFuncEvaluationsExceeded(Exception):
    pass


if __name__ == '__main__':
    # example code
    n = 5
    x = np.random.uniform(-2, 2, size=n)
    max_n_fvals = 5e4
    obj_func = lambda x: shubert_with_count(x, max_n_fvals)

    best_x, best_fval, history = \
        tabu_search(x, obj_func, n, step_size=1, n_sectors=5,
                si_thresh=10, sd_thresh=15, ssr_thresh=25,
                stm_size=26, mtm_size=4, ssr_factor=0.8, explored_thresh=10,
                epsilon=1e-9, max_n_fvals=max_n_fvals)
    
    print(best_fval, best_x)
