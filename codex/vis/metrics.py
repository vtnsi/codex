import math
import numpy as np

import statsmodels.api as sm
import statsmodels.regression.linear_model as sm_lin
import numpy as np


def standardized_proportion_per_interaction(counts, row, column):
    """
    Normalized computation of an interaction's proportion frequency.
    """
    n_jl = counts[row][column]
    c_j = len(counts[row])
    N = np.sum(counts[row])

    return (n_jl - N / c_j) / N


def proportion_per_interaction(counts, row, column):
    n_jl = counts[row][column]
    N = np.sum(counts[row])

    return n_jl / N


def slope_test(x, y):
    x_sm = sm.add_constant(x)
    y_sm = y
    model = sm_lin.OLS(endog=y_sm, exog=x_sm)
    results = model.fit()

    r = np.zeros_like(results.params)
    r = [1, 1]
    T_test = results.t_test(np.diag(r))
    return T_test

def standardized_proportion_frequency_bounds(N:int, c_i:int):
    upper_bound = (N-(N/c_i))/N
    lower_bound = (0-(N/c_i))/N

    return lower_bound, upper_bound #np.floor(lower_bound), np.ceil(upper_bound)

def standardized_proportion_frequency_bounds_iterative(N, counts):
    uppers = []
    lowers = []
    for counts_combo_i in counts:
        c_i = len(counts_combo_i)
        lower_candidate, upper_candidate = standardized_proportion_frequency_bounds(N, c_i)
        uppers.append(upper_candidate)
        lowers.append(lower_candidate)

    print("Upper bounds:", uppers)
    print("Lower bounds:", lowers)
    return np.min(lowers), np.max(uppers)
