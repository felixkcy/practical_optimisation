import numpy as np
from scipy import stats
from time import time

n_fvals = 0
best_fval_history = []
timestamps = []
start_time = 0


def recombination(X, method='discrete', n_offspring=140):
    if method == 'discrete':
        return discrete_recombination(X, n_offspring=n_offspring)

    elif method == 'global discrete':
        return global_discrete_recombination(X, n_offspring=n_offspring)

    elif method == 'intermediate':
        return intermediate_recombination(X, n_offspring=n_offspring)

    elif method == 'global intermediate':
        return global_intermediate_recombination(X, n_offspring=n_offspring)

    else:
        raise ValueError('unknown recombination method')


def discrete_recombination(X, n_offspring=140):
    population, n = X.shape
    offspring = np.zeros((n_offspring, n))
    for i in range(n_offspring):
        # randomly select 2 parents
        parent1, parent2 = np.random.randint(population, size=2)
        while parent1 == parent2:
            parent2 = np.random.randint(population)
        # randomly determine which parent contribute to which component
        coin_tosses = np.random.choice([True, False], n)

        # recombine
        offspring[i, coin_tosses] = X[parent1, coin_tosses]
        offspring[i, ~coin_tosses] = X[parent2, ~coin_tosses]

    return offspring


def global_discrete_recombination(X, n_offspring=140):
    population, n = X.shape
    offspring = np.zeros((n_offspring, n))

    for i in range(n_offspring):
        # randomly select parent for each component by balanced roulette wheel
        parents = np.random.randint(population, size=n)
        # recombine
        offspring[i, :] = X[parents, range(n)]

    return offspring


def intermediate_recombination(X, n_offspring=140, w=0.5):
    population, n = X.shape
    offspring = np.zeros((n_offspring, n))

    for i in range(n_offspring):
        # randomly select 2 parents
        parent1, parent2 = np.random.randint(population, size=2)
        while parent1 == parent2:
            parent2 = np.random.randint(population)

        # recombine by weighted average
        offspring[i, :] = w * X[parent1] + (1 - w) * X[parent2]

    return offspring


def global_intermediate_recombination(X, n_offspring=140, w=0.5):
    population, n = X.shape
    offspring = np.zeros((n_offspring, n))

    for i in range(n_offspring):
        for j in range(n):
            # randomly select 2 parents
            parent1, parent2 = np.random.randint(population, size=2)
            while parent1 == parent2:
                parent2 = np.random.randint(population)
            # take weighted average for that component
            offspring[i, j] = w * X[parent1, j] + (1 - w) * X[parent2, j]

    return offspring


def mutation(X, SP, tau, tau_prime, beta, alpha_indices, mutate_angles=True):
    n = X.shape[1]
    S = SP[:, :n].copy()    # standard deviations
    A = SP[:, n:].copy()    # rotation angles
    X_new = np.zeros_like(X)

    for i in range(X.shape[0]):
        # mutate sigma and alpha
        rv = np.random.randn(SP.shape[1] + 1)
        S[i] = S[i] * np.exp(tau_prime * rv[0] + tau * rv[1:n+1])
        if mutate_angles:
            A[i] = A[i] + beta * rv[n+1:]
        SP_new = np.concatenate((S, A), axis=-1)

        # mutate x
        # sample uncorrelated Gaussian
        dx = stats.multivariate_normal.rvs(np.zeros(n), np.diag(np.square(S[i])))
        if mutate_angles:
            # rotate dx by the n(n-1)/2 angles
            for alpha_i, alpha in enumerate(A[i]):
                # construct rotation matrix
                R = np.eye(n)
                ri, rj = alpha_indices[alpha_i]
                R[(ri, rj, ri, rj), (ri, rj, rj, ri)] = \
                    [np.cos(alpha), np.cos(alpha), -np.sin(alpha), np.sin(alpha)]
                # rotate dx
                dx = np.dot(R, dx)
        
        X_new[i] = X[i] + dx
    
    return X_new, SP_new


def evolution_strategy(obj_func, n, population=140, n_parents=20, s0=0.1,
                       x_recombination_method='discrete',
                       sp_recombination_method='intermediate',
                       select_parents=False, mutate_angles=True, epsilon=1e-12,
                       max_n_fvals=1e4):
    try:
        reset()

        X = np.random.uniform(-2, 2, size=(population, n))  # randomly sample in the space
        S0 = s0 * np.ones((population, n))      # initial std
        A0 = np.zeros((population, int(n*(n-1)/2)))       # initial rotation angles
        SP = np.concatenate((S0, A0), axis=-1)      # strategy parameters

        # control parameters
        tau = 1 / np.sqrt(2 * np.sqrt(n))
        tau_prime = 1 / np.sqrt(2 * n)
        beta = 0.0873

        alpha_indices = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]

        # assess initial population
        population_fvals = np.array([obj_func(x) for x in X])
        sorted_indices = np.argsort(population_fvals)
        best_x, best_fval = X[sorted_indices[0]], population_fvals[sorted_indices[0]]
        
        # save intermediate results for analysis
        history = {'X': [X], 'population_fvals': [population_fvals], 'SP': [SP],
                   'best_fval': [best_fval], 'convergence': -1}

        while n_fvals <= max_n_fvals:
            # select
            parents_indices = sorted_indices[:n_parents]
            X_parents = X[parents_indices]
            SP_parents = SP[parents_indices]
            fval_parents = population_fvals[parents_indices]

            if history['convergence'] == -1 and \
                    abs(min(fval_parents) - max(fval_parents)) < epsilon:
                # convergence criteria met
                history['convergence'] = n_fvals

            # recombine
            X = recombination(X_parents, method=x_recombination_method,
                              n_offspring=population)
            SP = recombination(SP_parents, method=sp_recombination_method,
                               n_offspring=population)

            # mutate
            X, SP = mutation(X, SP, tau, tau_prime, beta, alpha_indices,
                             mutate_angles=mutate_angles)

            # remove invalid solutions
            mask = np.all((X >= -2) & (X <= 2), axis=1)
            X = X[mask]
            SP = SP[mask]

            # evaluate
            population_fvals = np.array([obj_func(x) for x in X])

            if select_parents:      # mu + lambda selection scheme
                # include parents in new population
                X = np.concatenate((X, X_parents), axis=0)
                population_fvals = np.concatenate((population_fvals, fval_parents), axis=0)
                SP = np.concatenate((SP, SP_parents), axis=0)

            # sort population by fval for selection
            sorted_indices = np.argsort(population_fvals)
            if population_fvals[sorted_indices[0]] < best_fval:
                best_x, best_fval = \
                    X[sorted_indices[0]], population_fvals[sorted_indices[0]]

            # record for later analysis
            history['X'].append(X)
            history['population_fvals'].append(population_fvals)
            history['SP'].append(SP)
            history['best_fval'].append(best_fval)

    except MaxFuncEvaluationsExceeded:
        history['best_fval'] = best_fval_history.copy()
        history['timestamps'] = timestamps.copy()
        return best_x, best_fval, history

    history['best_fval'] = best_fval_history.copy()
    history['timestamps'] = timestamps.copy()
    return best_x, best_fval, history
    

def reset():
    global n_fvals, start_time
    n_fvals = 0
    start_time = time()
    best_fval_history.clear()
    timestamps.clear()


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
    max_n_fvals = 1e4
    obj_func = lambda x: shubert_with_count(x, max_n_fvals=max_n_fvals)

    best_x, best_fval, history = \
        evolution_strategy(obj_func, n, population=105, n_parents=15, s0=0.1,
                           x_recombination_method='discrete',
                           sp_recombination_method='global discrete',
                           select_parents=False, mutate_angles=False,
                           max_n_fvals=1e4)

    print(best_fval)