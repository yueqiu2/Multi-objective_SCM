from a6_re_env import InvManagementDiv
import a9b_

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer

import matplotlib.pyplot as plt


###########
# PRIMARY #
###########

# Inventory environment
config = {"demand_dist": "uniform",
         "noisy_delay": True,
         "noisy_delay_threshold": 0.5,
         "time_dependency": True,
         "prev_demand": True,
         }
env = InvManagementDiv(config=config)

# Problem setting
inv_problem = a9b_.problem_setting(num_nodes=3, length=3, trans_type=3, num_periods=20)
inv_problem = a9b_.problem_setting(num_nodes=3, length=6, trans_type=3, num_periods=20)
inv_problem = a9b_.problem_setting(num_nodes=3, length=9, trans_type=3, num_periods=20)
# NN evaluation
def hp_opt_nsga2(hyperparameter):
    """
    :param hyperparameter: tuple
    :return: performance indicator
    """

    # unpack hyperparameters
    pop_size, n_gen, n_offsprings = hyperparameter

    # function evaluation budget
    FE = pop_size + n_offsprings * (n_gen - 1)
    if FE > 3e4:
        return 10  # large number
    else:
        # run the primary optimization - optimize the NN parameter via MOEAs
        res_nsga2 = a9b_.run_nsga2(problem=inv_problem,
                                   pop_size=pop_size,
                                   n_gen=n_gen,
                                   n_offsprings=n_offsprings)

        # evaluate the NN obtained by certain set of hyperparameter - hypervolume
        final_hv = a9b_.hypervolume(res_nsga2)

        # should maximize hypervolume, but use gp_minimize in meta optimization
        return -final_hv

def hp_opt_age(hyperparameter):
    """
    :param hyperparameter: tuple
    :return: performance indicator
    """

    # unpack hyperparameters
    pop_size, n_gen = hyperparameter

    # function evaluation budget
    FE = pop_size * n_gen
    if FE > 3e4:
        return 10  # large number
    else:
        # run the primary optimization - optimize the NN parameter via MOEAs
        res_age = a9b_.run_age(problem=inv_problem,
                               pop_size=pop_size,
                               n_gen=n_gen,)

        # evaluate the NN obtained by certain set of hyperparameter - hypervolume
        final_hv = a9b_.hypervolume(res_age)

        # should maximize hypervolume, but use gp_minimize in meta optimization
        return -final_hv

########
# META #
########

# range of hp
hp_space_nsga2 = [
    Integer(100, 300, name="pop_size"),
    Integer(50, 200, name="n_gen"),
    Integer(50, 300, name="n_offsprings")
]

hp_space_age = [
    Integer(100, 300, name="pop_size"),
    Integer(50, 200, name="n_gen"),
]

def progress(res):
    """
    Prints the iteration number as optimization progresses.
    """
    print(f"Bayesian Optimization Iteration #{len(res.func_vals)}")

# bayesian optimization
def BO_nsga2(n_calls):

    result = gp_minimize(hp_opt_nsga2, hp_space_nsga2, n_calls=n_calls, random_state=0, callback=[progress])

    # best hyperparameter
    hp = result.x

    # hypervolume of each bayesian iteration (negative here)
    hv = result.func_vals

    # convergence plot
    iterations = range(1, n_calls + 1)
    plt.plot(iterations, -hv, color='blue',label='Hypervolume', linestyle='-', marker='o')
    plt.scatter(iterations, -hv, color='blue',  facecolor="none", marker='o', label='Hypervolume')
    plt.xticks(iterations)
    plt.yticks(np.arange(0, 1.2, 0.1))
    plt.xlabel('BO Iteration NSGA-II')
    plt.ylabel('Hypervolume')
    plt.show()

    return hp

def BO_age(n_calls):

    result = gp_minimize(hp_opt_age, hp_space_age, n_calls=n_calls, random_state=0, callback=[progress])

    # best hyperparameter
    hp = result.x

    # hypervolume of each bayesian iteration (negative)
    hv = result.func_vals

    # convergence plot
    iterations = range(1, n_calls + 1)
    plt.plot(iterations, -hv, color='blue', label='Hypervolume', linestyle='-', marker='o')
    plt.scatter(iterations, -hv, color='blue', facecolor="none", marker='o', label='Hypervolume')
    plt.xticks(iterations)
    plt.xlabel('BO Iteration AGE-MOEA')
    plt.ylabel('Hypervolume')
    plt.show()

    return hp


#%% Test run

# result - best hp
# # NSGA-II
# hp_nsga2 = BO_nsga2(10)  # best hp for nsga2
# print(hp_nsga2)
# pop_size, n_gen, n_offsprings = hp_nsga2
# res_nsga2_best = a9b_.run_nsga2(problem=inv_problem,
#                           pop_size=pop_size,
#                           n_gen=n_gen,
#                           n_offsprings=n_offsprings)


# NSGA-II
hp_age = BO_age(10)  # best hp of age
print(hp_age)
pop_size, n_gen = hp_age
res_age_best = a9b_.run_age(problem=inv_problem,
                            pop_size=pop_size,
                            n_gen=n_gen,)


# get the graph for both
# a9b_.convergence_PF_plot_total(res_nsga2_best, res_age_best)

