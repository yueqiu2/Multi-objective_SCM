from a6_re_env import InvManagementDiv
from a7_NN import Net

import numpy as np
import torch
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2


from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from pymoo.optimize import minimize
from pymoo.termination import get_termination

from pymoo.indicators.hv import Hypervolume

from sklearn.preprocessing import MinMaxScaler

# Inventory environment
config={
        #"demand_dist": "uniform",
        #"noisy_delay": False,
        #"noisy_delay_threshold": 0.5,
        }
env = InvManagementDiv(config=config)

###################
# Problem setting #
###################
def unflatten_params(vec, dict):
    """
    :param dict: original dict, for structure ref
    :return: dict
    """

    # dictionary for policy
    policy = {}

    j = 0
    for k, v in dict.items():
        # CREATE TENSOR: torch.from_numpy()
        v_size = np.prod(v.shape)
        new_tensor = torch.from_numpy(vec[j:j + v_size].reshape(v.shape))
        # replace old tensor in the key with new tensor
        policy[k] = new_tensor
        j = j + v_size

    return policy

class InvProblem(Problem):
    # statement
    def __init__(self, dim, num_nodes, length, trans_type, num_periods):
        self.dim = dim
        self.num_nodes = num_nodes
        self.length = length
        self.trans_type = trans_type
        self.num_periods = num_periods

        # bounds
        xl = np.ones(dim) * -10
        xu = np.ones(dim) * 10

        super().__init__(n_var=dim,
                         n_obj=2,
                         n_constr=0,
                         xl=xl,
                         xu=xu)
    # problem
    def _evaluate(self, x, out, *args, **kwargs):

        # initialize the policy net
        policy_net = Net(self.num_nodes, self.length, self.trans_type)
        # dictionary structure
        dict_struc = policy_net.state_dict()

        # recording
        f1_list = []
        f2_list = []

        # TODO assume x is the whole for x
        for xi in x:

            # get the dict from individual flat array
            NN_param = unflatten_params(xi, dict_struc)
            # load it
            policy_net.load_state_dict(NN_param)

            # initialize state
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float)

            # calculate the objectives for one episode
            for t in range(self.num_periods):

                # get action
                action = policy_net(state)

                # convert tensor to array
                reorder_array = action[0].detach().cpu().numpy()
                trans_choice_array = action[1].detach().cpu().numpy()
                # put it back to tuple
                action = (reorder_array[0], trans_choice_array)

                # state transition
                next_state, rewards, emissions, done, info = env.step(action)
                # update state
                state = torch.tensor(next_state, dtype=torch.float)

                if done:
                    break

            f1_ = rewards
            f2_ = emissions

            f1_list.append(f1_)
            f2_list.append(f2_)

        f1_list = np.array(f1_list)
        f2_list = np.array(f2_list)

        out['F'] = np.column_stack([f1_list, f2_list])

def problem_setting(num_nodes, length, trans_type, num_periods):

    # NN structure
    policy_net = Net(num_nodes, length, trans_type)
    net_struct = policy_net.state_dict()

    # get dimensionality by setting
    flattened_params = [value.detach().numpy().flatten() for value in net_struct.values()]
    flattened_params = np.concatenate(flattened_params)
    dim = flattened_params.shape[0]

    # problem instantiation
    inv_problem = InvProblem(dim=dim,
                             num_nodes=num_nodes,
                             length=length,
                             trans_type=trans_type,
                             num_periods=num_periods)

    return inv_problem

##############
# Algorithms #
##############
def run_nsga2(problem, pop_size, n_gen, n_offsprings):

    # NSGA-II
    algorithms_NSGA2 = NSGA2(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    # termination condition
    termination = get_termination('n_gen', n_gen)

    # run algorithm
    res = minimize(problem, algorithms_NSGA2,
                   termination=termination, seed=1, save_history=True, verbose=True)

    return res

def run_age(problem, pop_size, n_gen):

    # AGEMOEA
    algorithms_AGE = AGEMOEA(pop_size=pop_size)
    # termination condition
    termination = get_termination('n_gen', n_gen)

    # run algorithm
    res = minimize(problem, algorithms_AGE,
                   termination=termination, seed=1, save_history=True, verbose=True)

    return res

######
# hv #
######

# Calculate hv
def hypervolume(res):

    # final front
    F = res.F

    # read the history record
    hist = res.history
    n_evals = []             # corresponding number of function evaluations\
    hist_F = []              # the objective space values in each generation

    for algo in hist:
        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)
        # retrieve the optimum from the algorithm
        opt = algo.opt
        hist_F.append(opt.get("F"))


    # Hypervolume: scale the PF to [0~1]

    hist_F = np.array(hist_F, dtype=object)
    # get num of individuals in each generation front
    shape = [array.shape[0] for array in hist_F]

    # flatten the list
    hist_F_flat = np.vstack(hist_F)

    # normalize the phenotype space
    scalar = MinMaxScaler()
    hist_F_flat_norm = scalar.fit_transform(hist_F_flat)

    # unflatten the normalized phenotype space
    hist_F_norm = np.split(hist_F_flat_norm, np.cumsum(shape)[:-1])
    hist_F_norm = np.array(hist_F_norm, dtype=object)
    shape_norm = [array.shape for array in hist_F_norm]

    # Get the normalized final front
    F_norm = hist_F_norm[-1]

    # ref point should be worse than all PF
    ref_point = np.array([1.1, 1.1])

    approx_ideal = F_norm.min(axis=0)  # expected to be [0 0]
    approx_nadir = F_norm.max(axis=0)  # expected to be less than [1 1]

    # hypervolume
    metric = Hypervolume(ref_point=ref_point,
                         norm_ref_point=False,
                         zero_to_one=True,
                         ideal=approx_ideal,
                         nadir=approx_nadir)

    hv = [metric.do(_F) for _F in hist_F_norm]

    # return the final hypervolume
    return hv[-1]

def hypervolume_total(res_nsga2, res_age):

    # final front
    F_nsga2 = res_nsga2.F
    F_age   = res_age.F

    # read the history record
    hist_nsga2 = res_nsga2.history
    n_evals_nsga2 = []             # corresponding number of function evaluations
    hist_F_nsga2 = []              # the objective space values in each generation

    for algo in hist_nsga2:
        # store the number of function evaluations
        n_evals_nsga2.append(algo.evaluator.n_eval)
        # retrieve the optimum from the algorithm
        opt = algo.opt
        hist_F_nsga2.append(opt.get("F"))

    # read the history record
    hist_age = res_age.history
    n_evals_age = []  # corresponding number of function evaluations\
    hist_F_age = []  # the objective space values in each generation

    for algo in hist_age:
        # store the number of function evaluations
        n_evals_age.append(algo.evaluator.n_eval)
        # retrieve the optimum from the algorithm
        opt = algo.opt
        hist_F_age.append(opt.get("F"))


    # Hypervolume: scale the PF to [0~1]

    # stack F from both algorithms together
    hist_F_nsga2 = np.array(hist_F_nsga2, dtype=object)
    hist_F_age = np.array(hist_F_age, dtype=object)

    n_gen_nega2 = len(hist_F_nsga2)
    n_gen_age   = len(hist_F_age)

    hist_F_total = np.concatenate((hist_F_nsga2, hist_F_age))
    hist_F_total = np.array(hist_F_total, dtype=object)
    # get num of individuals in each generation front
    shape = [len(array) for array in hist_F_total]

    # flatten the total list
    hist_F_flat = np.vstack(hist_F_total)

    # normalize the phenotype space, for both algorithms
    scalar = MinMaxScaler()
    hist_F_flat_norm = scalar.fit_transform(hist_F_flat)

    # unflatten the normalized phenotype space, for both algorithms
    hist_F_norm = np.split(hist_F_flat_norm, np.cumsum(shape)[:-1])
    hist_F_norm = np.array(hist_F_norm, dtype=object)
    shape_norm = [array.shape for array in hist_F_norm]

    # break the total F
    hist_F_norm_nsga2 = hist_F_norm[:n_gen_nega2]
    hist_F_norm_age   = hist_F_norm[n_gen_nega2:]

    assert(len(hist_F_nsga2) == n_gen_nega2)
    assert(len(hist_F_age) == n_gen_age)

    # Get the normalized final front
    F_norm_nsga2 = hist_F_norm_nsga2[-1]
    F_norm_age   = hist_F_norm_age[-1]

    # ref point should be worse than all PF
    ref_point = np.array([1.1, 1.1])


    # hv for nsga2
    approx_ideal1 = F_norm_nsga2.min(axis=0)  # expected to be equal or greater than [0 0]
    approx_nadir1 = F_norm_nsga2.max(axis=0)  # expected to be less than [1 1]

    # hypervolume
    metric_nsga2 = Hypervolume(ref_point=ref_point,
                               norm_ref_point=False,
                               zero_to_one=True,
                               ideal=approx_ideal1,
                               nadir=approx_nadir1)

    hv_nsga2 = [metric_nsga2.do(_F) for _F in hist_F_norm_nsga2]

    # hv for agemoea
    approx_ideal2 = F_norm_age.min(axis=0)  # expected to be equal to [0 0]
    approx_nadir2 = F_norm_age.max(axis=0)  # expected to be less than [1 1]

    # hypervolume
    metric_age = Hypervolume(ref_point=ref_point,
                               norm_ref_point=False,
                               zero_to_one=True,
                               ideal=approx_ideal2,
                               nadir=approx_nadir2)

    hv_age = [metric_age.do(_F) for _F in hist_F_norm_age]

    return hv_nsga2[-1], hv_age[-1]



# Plotting
def convergence_PF_plot(res):

    # final front
    F = res.F

    # read the history record
    hist = res.history
    n_evals = []             # corresponding number of function evaluations\
    hist_F = []              # the objective space values in each generation

    for algo in hist:
        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)
        # retrieve the optimum from the algorithm
        opt = algo.opt
        hist_F.append(opt.get("F"))


    # Hypervolume: scale the PF to [0~1]

    hist_F = np.array(hist_F, dtype=object)
    # get num of individuals in each generation front
    shape = [array.shape[0] for array in hist_F]

    # flatten the list
    hist_F_flat = np.vstack(hist_F)

    # normalize the phenotype space
    scalar = MinMaxScaler()
    hist_F_flat_norm = scalar.fit_transform(hist_F_flat)

    # unflatten the normalized phenotype space
    hist_F_norm = np.split(hist_F_flat_norm, np.cumsum(shape)[:-1])
    hist_F_norm = np.array(hist_F_norm, dtype=object)
    shape_norm = [array.shape for array in hist_F_norm]

    # Get the normalized final front
    F_norm = hist_F_norm[-1]

    # ref point should be worse than all PF
    ref_point = np.array([1.1, 1.1])

    approx_ideal = F_norm.min(axis=0)  # expected to be [0 0]
    approx_nadir = F_norm.max(axis=0)  # expected to be less than [1 1]

    # hypervolume
    metric = Hypervolume(ref_point=ref_point,
                         norm_ref_point=False,
                         zero_to_one=True,
                         ideal=approx_ideal,
                         nadir=approx_nadir)

    hv = [metric.do(_F) for _F in hist_F_norm]

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # phenotype space
    axes[0].scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    axes[0].set_xlabel('Profit')
    axes[0].set_ylabel('Emission')
    axes[0].set_title('Objective')

    # hypervolume
    axes[1].scatter(n_evals, hv, facecolor="none", edgecolor='black', marker="p")
    axes[1].set_xlabel('Function Evaluations')
    axes[1].set_ylabel('Hypervolume')
    y_ticks = np.arange(0, 1.1, 0.1)
    axes[1].set_yticks(y_ticks)
    axes[1].set_title('Convergence')

    plt.tight_layout()
    plt.show()

    # return the final hypervolume
    return hv[-1]

def convergence_PF_plot_total(res_nsga2, res_age):

    # final front
    F_nsga2 = res_nsga2.F
    F_age   = res_age.F

    # read the history record
    hist_nsga2 = res_nsga2.history
    n_evals_nsga2 = []             # corresponding number of function evaluations
    hist_F_nsga2 = []              # the objective space values in each generation

    for algo in hist_nsga2:
        # store the number of function evaluations
        n_evals_nsga2.append(algo.evaluator.n_eval)
        # retrieve the optimum from the algorithm
        opt = algo.opt
        hist_F_nsga2.append(opt.get("F"))

    # read the history record
    hist_age = res_age.history
    n_evals_age = []  # corresponding number of function evaluations\
    hist_F_age = []  # the objective space values in each generation

    for algo in hist_age:
        # store the number of function evaluations
        n_evals_age.append(algo.evaluator.n_eval)
        # retrieve the optimum from the algorithm
        opt = algo.opt
        hist_F_age.append(opt.get("F"))

    # Hypervolume: scale the PF to [0~1]

    # stack F from both algorithms together
    hist_F_nsga2 = np.array(hist_F_nsga2, dtype=object)
    hist_F_age = np.array(hist_F_age, dtype=object)

    n_gen_nega2 = len(hist_F_nsga2)
    n_gen_age = len(hist_F_age)

    hist_F_total = np.concatenate((hist_F_nsga2, hist_F_age))
    hist_F_total = np.array(hist_F_total, dtype=object)
    # get num of individuals in each generation front
    shape = [len(array) for array in hist_F_total]

    # flatten the total list
    hist_F_flat = np.vstack(hist_F_total)

    # normalize the phenotype space, for both algorithms
    scalar = MinMaxScaler()
    hist_F_flat_norm = scalar.fit_transform(hist_F_flat)

    # unflatten the normalized phenotype space, for both algorithms
    hist_F_norm = np.split(hist_F_flat_norm, np.cumsum(shape)[:-1])
    hist_F_norm = np.array(hist_F_norm, dtype=object)
    shape_norm = [array.shape for array in hist_F_norm]

    # break the total F
    hist_F_norm_nsga2 = hist_F_norm[:n_gen_nega2]
    hist_F_norm_age = hist_F_norm[n_gen_nega2:]

    assert (len(hist_F_nsga2) == n_gen_nega2)
    assert (len(hist_F_age) == n_gen_age)

    # Get the normalized final front
    F_norm_nsga2 = hist_F_norm_nsga2[-1]
    F_norm_age = hist_F_norm_age[-1]

    # ref point should be worse than all PF
    ref_point = np.array([1.1, 1.1])

    # hv for nsga2
    approx_ideal1 = F_norm_nsga2.min(axis=0)  # expected to be equal or greater than [0 0]
    approx_nadir1 = F_norm_nsga2.max(axis=0)  # expected to be less than [1 1]

    # hypervolume
    metric_nsga2 = Hypervolume(ref_point=ref_point,
                               norm_ref_point=False,
                               zero_to_one=True,
                               ideal=approx_ideal1,
                               nadir=approx_nadir1)

    hv_nsga2 = [metric_nsga2.do(_F) for _F in hist_F_norm_nsga2]

    # hv for agemoea
    approx_ideal2 = F_norm_age.min(axis=0)  # expected to be equal to [0 0]
    approx_nadir2 = F_norm_age.max(axis=0)  # expected to be less than [1 1]

    # hypervolume
    metric_age = Hypervolume(ref_point=ref_point,
                               norm_ref_point=False,
                               zero_to_one=True,
                               ideal=approx_ideal2,
                               nadir=approx_nadir2)

    hv_age = [metric_age.do(_F) for _F in hist_F_norm_age]

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # phenotype space
    axes[0].scatter(F_nsga2[:, 0], F_nsga2[:, 1], s=30, label='NSGA-II', facecolors='none', edgecolors='blue')
    axes[0].scatter(F_age[:, 0], F_age[:, 1], s=30, label='AGE-MOEA', facecolors='none', edgecolors='red')

    axes[0].set_xlabel('Profit')
    axes[0].set_ylabel('Emission')
    axes[0].set_title('Objective')

    # hypervolume
    axes[1].scatter(n_evals_nsga2, hv_nsga2, label='NSGA-II', facecolor="none", edgecolor='blue', marker="p")
    axes[1].scatter(n_evals_age, hv_age, label='AGE-MOEA', facecolor="none", edgecolor='red', marker="p")
    axes[1].set_xlabel('Function Evaluations')
    axes[1].set_ylabel('Hypervolume')
    y_ticks = np.arange(0, 1.2, 0.1)
    axes[1].set_yticks(y_ticks)
    axes[1].set_title('Convergence')

    axes[0].legend(loc='best')
    axes[1].legend(loc='best')

    plt.tight_layout()
    plt.show()

    # Extreme points for AGE-MOEA
    age_max_profit = round(F_age[:, 0].min())
    age_min_emission = round(F_age[:, 1].min())
    age_min_profit = round(F_age[:, 0].max())
    age_max_emission = round(F_age[:, 1].max())

    # Annotation of extreme points
    # Extreme points for NSGA-II
    nsga2_max_profit = round(F_nsga2[:, 0].min())
    nsga2_min_emission = round(F_nsga2[:, 1].min())
    nsga2_min_profit = round(F_nsga2[:, 0].max())
    nsga2_max_emission = round(F_nsga2[:, 1].max())


    print(hv_nsga2[-1])
    print(np.array([age_max_profit, age_min_profit]))
    print(np.array([age_min_emission, age_max_emission]))

    print(hv_age[-1])
    print(np.array([nsga2_max_profit, nsga2_min_profit]))
    print(np.array([nsga2_min_emission, nsga2_max_emission]))


#%% Test run
inv_problem = problem_setting(num_nodes=3, length=3, trans_type=3, num_periods=20)
res_nsga2 = run_nsga2(problem=inv_problem, pop_size=260, n_gen=128, n_offsprings=200)
res_age = run_age(problem=inv_problem, pop_size=225, n_gen=108)
# convergence_PF_plot(res_nsga2)
convergence_PF_plot_total(res_nsga2, res_age)