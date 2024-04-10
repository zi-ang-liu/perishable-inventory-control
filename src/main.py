from gurobipy import GRB, Model, quicksum
import numpy as np
from problem_dynamics import build_dynamics_fifo
from lp_solver import lp_solver
from value_iteration import value_iteration
import matplotlib.pyplot as plt
import seaborn as sns

def dynamics_transfer(dynamics):
    # calculate r(s,a) and p(s'|s,a)
    r = np.zeros((len(state_space), len(action_space)))
    p = np.zeros((len(state_space), len(action_space), len(state_space)))

    # transfer state to index
    state_to_index = {}
    for i, state in enumerate(state_space):
        state_to_index[state] = i

    # calculate r(s,a) and p(s'|s,a)
    for state in state_space:
        for action in action_space:
            for (next_state, reward), prob in dynamics[state, action].items():
                r[state_to_index[state], action] += reward * prob
                p[state_to_index[state], action,
                    state_to_index[next_state]] += prob
                
    # assert sum of p(s'|s,a) = 1
    for state in state_space:
        for action in action_space:
            assert abs(sum(p[state_to_index[state], action, :]) - 1) < 1e-6

    return r,p, state_to_index

if __name__ == '__main__':

    # set parameters for problem
    life_time = 2
    lead_time = 0
    unit_lost_cost = 5
    unit_hold_cost = 1
    unit_perish_cost = 7
    unit_order_cost = 3
    max_order = 5
    mean_demand = 5
    cv = 0.5

    # set gamma
    gamma = 0.99

    parameters = {
        'life_time': life_time,
        'lead_time': lead_time,
        'unit_lost_cost': unit_lost_cost,
        'unit_hold_cost': unit_hold_cost,
        'unit_perish_cost': unit_perish_cost,
        'unit_order_cost': unit_order_cost,
        'max_order': max_order,
        'mean_demand': mean_demand,
        'cv': cv
    }

    dynamics, state_space, action_space, value, policy = build_dynamics_fifo(
        parameters)
    
    # # transfer dynamics to r(s,a) and p(s'|s,a)
    # r, p, state_to_index = dynamics_transfer(dynamics)

    # # solve LP
    # model = lp_solver(r, p, gamma)

    # state value
    # for state in state_space:
    #     value[state] = model.getVarByName(
    #         'v_{}'.format(state_to_index[state])).x


    # value iteration
    value, policy = value_iteration(dynamics, state_space, action_space,
                        value, policy, theta=1e-5, gamma=0.99)


    for state in state_space:
        best_value = -np.inf
        for action in action_space:
            value_temp = sum([prob * (reward + gamma * value[next_state])
                             for (next_state, reward), prob in dynamics[state, action].items()])
            if value_temp > best_value:
                best_value = value_temp
                policy[state] = action


    # 2-dimensional plot
    if life_time + lead_time == 2:
        # plot policy
        ax = sns.heatmap(policy, annot=True, fmt="d")
        ax.invert_yaxis()
        plt.title('Policy')
        plt.show()

        # plot state value
        ax = sns.heatmap(value, annot=True, fmt=".2f")
        ax.invert_yaxis()
        plt.title('State Value')
        plt.show()