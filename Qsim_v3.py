#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:51:40 2024

@author: junaidrehman
"""

import networkx as nx
from itertools import combinations, groupby
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import clear_output, display
import random


# environment and simulation setting
class QuantumRoutingAgent:
    def __init__(self, graph, start_node, destination,\
                 alpha=0.1, gamma=0.9, epsilon=0.5, mode = 3, episodes = 1000, qtable = None, save_training = False):
        self.G = graph
        self.start_node = start_node
        self.destination = destination
        self.curr_node = start_node
        self.path = [start_node]
        self.optimal_path = []
        self.optimal_extended_path = []
        self.highest_reward = -float('inf')
        self.best_fidelity = None
        self.best_rate = None
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.done = False
        self.current_reward = 0
        self.purif_decision = 0
        self.path_extended = [self.start_node] # path + purifications
        self.episodes = episodes
        self.mode = mode
        self.curr_fid = 1
        self.save_figure = save_training
        if self.mode == 1: # only maximize rate
            self.weight = [0, 10, 0] # fidelity, rate, path length
        elif self.mode == 2: # balanced 1:1 is balanced because of non-linear relation between rate dec and fidelity inc
            self.weight = [4, 6, 0]
        elif self.mode == 3: # only maximize fidelity
            self.weight = [10, 0, 0]
        else: 
            print(f"Incorrect mode {mode}, setting mode = 2 (balanced)")
            self.mode = 2
            self.weight = [4, 6, 0]
        self.aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
        
        # Q-table as a dictionary of state-action pairs
        if qtable is None:  # Q-table not supplied
            self.qtable_name = None
            self.Q_table = {}
            self.example_name = None
        else:  # Q-table supplied
            self.example_name = qtable
            qtable = f"qtable-qsim/qtable-{qtable}.npy"
            
            try:
                self.Q_table = np.load(qtable, allow_pickle=True).item()
                if not isinstance(self.Q_table, dict):
                    raise ValueError("Loaded Q-table is not a dictionary.")
            except Exception as e:
                print(f"Error loading Q-table from {qtable}: {e}, generating a new one.")
                self.Q_table = {}
                
        
    def reset(self):
        self.curr_node = self.start_node
        self.path = [self.start_node]
        self.path_extended = [self.start_node]
        self.done = False
        self.purif_decision = 0
        self.epsilon = max(0.01, self.epsilon * 0.9995)
        self.curr_fid = 1


    def _get_state(self):
        # Represent state as current node, swap decisions, and purification decisions
        return int(self.curr_node), self.purif_decision, self.mode

    def get_action(self, state):
        curr_node = state[0]

        # Get neighbors and sort by fidelity
        neighbors = self.G[curr_node]
        sorted_neighbors = sorted(neighbors.items(),
                                  key=lambda x: self.G[curr_node][x[0]]['fidelity'],
                                  reverse=True)
        sorted_neighbors = [s for s in sorted_neighbors if s[0] not in self.path]
        #print(self.path, sorted_neighbors)


        # Add movements to neighbors
        action_vector = [neighbor for neighbor, _ in sorted_neighbors]
        
        if len(self.path) > 1 and len(action_vector) > 0:
            # print(curr_node, self.path)
        #     print(f"**** {self.path_extended} curr_node: {curr_node}, previous node {self.path[-2]}\n purif_decision: {self.purif_decision}, \
        # allowed rounds: {self.G[self.path[-2]][curr_node]['pur_round']}")
            if self.purif_decision < self.G[self.path[-2]][curr_node]['pur_round']:
                action_vector.append(curr_node)
        # print(self.path, self.path_extended)
        if curr_node == self.destination and self.purif_decision < self.G[self.path[-2]][curr_node]['pur_round']:
            action_vector = [curr_node, -1] # adding stop signal as an option
        elif curr_node == self.destination:
            action_vector = [-1]

        # print(f"action vector: {action_vector}\n********")
        #print(self.path, curr_node, sorted_neighbors)
        #if len(action_vector) == 0 or self.curr_fid<0.5: # got in a loop, or reached to a bad fidelity
        if len(action_vector) == 0: # got in a loop, or reached to a bad fidelity
            self.done = True
            self.path.append(-1)
            self.path_extended.append(-1)
            return -1, [-1]
        # Epsilon-greedy strategy: explore or exploit
        if np.random.rand() < self.epsilon:
            chosen_action = np.random.choice(action_vector)
            # if chosen_action == curr_node:
            #     self.purif_decision += 1
            return chosen_action, action_vector  # Exploration (random action)
        else:
            # Exploitation (choose the best action based on Q-table)
            q_values = [self.Q_table.get((state, action), 0) for action in action_vector]
            max_q_value = max(q_values)
            best_action = np.random.choice([action for action, q_value in zip(action_vector, q_values) if q_value == max_q_value])
            # if best_action == curr_node:
            #     self.purif_decision += 1
            return best_action, action_vector

    def get_future_action(self, state):
        if state[0] == -1:
            return [-1]
        curr_node = state[0]

        # Get neighbors and sort by fidelity
        neighbors = self.G[curr_node]
        sorted_neighbors = sorted(neighbors.items(),
                                  key=lambda x: self.G[curr_node][x[0]]['fidelity'],
                                  reverse=True)
        sorted_neighbors = [s for s in sorted_neighbors if s[0] not in self.path]
        #print(self.path, sorted_neighbors)


        # Add movements to neighbors
        action_vector = [neighbor for neighbor, _ in sorted_neighbors]
        
        if len(self.path) > 1 and len(action_vector) > 0:
            if self.purif_decision < self.G[self.path[-2]][curr_node]['pur_round']:
                action_vector.append(curr_node)
        if curr_node == self.destination and self.purif_decision < self.G[self.path[-2]][curr_node]['pur_round']:
            action_vector = [curr_node, -1] # adding stop signal as an option
        elif curr_node == self.destination:
            action_vector = [-1]

        if len(action_vector) == 0: # got in a loop, or reached to a bad fidelity
            return [-1]
        return action_vector

    def step(self):
        
        curr_state = self._get_state()
        curr_node = curr_state[0]
        # print(curr_state)
        # possible_actions = self.get_action(curr_state)
        action, possible_actions = self.get_action(curr_state)
        #print(possible_actions)
        # action = self.choose_action(possible_actions, curr_state)
        if action != None:
            if not self.done:
                if action != curr_node:
                    self.path.append(int(action))
                    self.curr_node = action
                    self.path_extended.append(int(action))
                    self.purif_decision = 0
                    #print("******", action, curr_node, self.path_extended, self.path)
                else:
                    self.path_extended.append(int(action))
                    self.purif_decision += 1
                

        
            # Calculate reward after taking the action
            reward = self.calculate_reward(self.path)
            self.reward = reward
    
            # Update Q-table using the Bellman equation
            next_state = self._get_state()
            action_taken = action
            possible_future_actions = self.get_future_action(next_state)
            max_future_q = max([self.Q_table.get((next_state, a), 0) for a in possible_future_actions])
            current_q = self.Q_table.get((curr_state, action_taken), 0)
            updated_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
#             if curr_node == 19 and action == 7:
#                 print("19: ", current_q, max_future_q, updated_q)
            # print(current_q, updated_q)
            # Update Q-table

            self.Q_table[(curr_state, action_taken)] = updated_q
            
        else:
            reward = 0
            self.reward = reward

        


    def calculate_reward(self, path):
        fid_path = []
        rate_path = []
        if self.done and path[-2] != self.destination: # destination not reached
            #print("giving -10*len(path)")
            return 0
        # elif path[-1] != -1: # intermediate
        #     return  (self.mode - 2)*self.purif_decision - len(path)
    
        FF = 1
        for ii in range(len(path) - 1):
            
            # Validate the edge
            if path[ii+1] != -1:
                F = self.G[path[ii]][path[ii + 1]]['fidelity']
                R = self.G[path[ii]][path[ii + 1]]['rate']
                pur_rounds = self.path_extended.count(path[ii+1])-1
                for pr in range(pur_rounds):
                    R = 0.5*R*(F**2 + (1 - F)**2) # halved due to meas. then reduced due to probability
                    F = F**2/(F**2 + (1 - F)**2) # purified fidelity
                    
    
    
                fid_path.append(F)
                rate_path.append(R)
                FF = 0.25 + (4 * FF - 1) * (4 * fid_path[-1] - 1) / 12
                

        swap_prob = np.prod([self.G.nodes[self.path[ii]]["swap_success"] for ii in range(1, len(path)-1)])
        # print(f"path: {self.path}, swap_prob: {[self.G.nodes[self.path[ii]]['swap_success'] for ii in range(1, len(path)-1)]}, {swap_prob}")
        final_fidelity = FF # fidelity after all swapped links with purification
        if path[-1] != -1: # intermediate, just check the fidelity and return accordingly
            self.curr_fid = FF
            if FF <= 0.5:
                return -10
            else:
                return (self.mode - 2)*self.purif_decision - len(path)
        # rate calculation
        swap_probs = [self.G.nodes[self.path[ii]]["swap_success"] \
                      for ii in range(1, len(path)-1)]
        
        #print(path, rate_path, swap_probs)
        curr_rate = rate_path[0]
        #print(swap_probs, rate_path, curr_rate)
        for ii in range(len(swap_probs)-1):
            curr_rate = swap_probs[ii]*np.min([rate_path[ii+1], curr_rate])
        final_rate = curr_rate # rate after all swapping

        if final_fidelity <= 0.5: # no reward for fidelity below 0.5
            reward = -10
        else:
            reward = np.dot(self.weight, [final_fidelity, final_rate, \
                                           -len(self.path)])
        if reward > self.highest_reward:
            self.highest_reward = reward
            self.optimal_path = path
            self.best_fidelity = final_fidelity
            self.best_rate = final_rate
            self.optimal_extended_path = self.path_extended
                

        
        return reward

    def run_simulation(self, steps):
        episodes = self.episodes
        ep_rewards = []
        for ep in range(episodes):
            self.reset()  # Reset environment for new episode
            while not self.done:
            # for _ in range(10): # running for 10 steps
                self.step()
                if self.path[-1] == -1 or len(self.path) > 50:
                    self.done = True
                    # print("done signal reached")
                    #self.step()  # For reward calculation
            # if ep % (np.round(episodes/10)) == 0:
            #     print(f"Episode {ep}: Path - {self.path}, {self.path_extended}, {self.reward}")
            STATS_EVERY = int(np.round(episodes/100))
            ep_rewards.append(self.reward)
            if 0: # never going here
                average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
                self.aggr_ep_rewards['ep'].append(ep)
                self.aggr_ep_rewards['avg'].append(average_reward)
                self.aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
                self.aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))

        
        if self.save_figure:
            plt.figure(figsize=(3.5, 2.5))  # Set the figure size
            plt.plot(self.aggr_ep_rewards['ep'], self.aggr_ep_rewards['avg'], 
             label=r"Average Rewards", color='blue', linestyle='-', linewidth=2)
    
            # Plot max rewards
            plt.plot(self.aggr_ep_rewards['ep'], self.aggr_ep_rewards['max'], 
                     label=r"Max Rewards", color='red', linestyle='--', linewidth=2)
            
            # Plot min rewards
            plt.plot(self.aggr_ep_rewards['ep'], self.aggr_ep_rewards['min'], 
                     label=r"Min Rewards", color='green', linestyle='-.', linewidth=2)
            
            # Customizing the plot
            plt.xlabel(r"Episodes")
            plt.ylabel(r"Rewards")
            #plt.title(r"\textbf{Episode Rewards}")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend(loc='lower right')
            plt.tight_layout()  # Adjust layout to avoid clipping
            plt.xlim([0, self.episodes])
            plt.ylim([0, 10])
            # # Save the plot as a high-resolution image
            # plt.savefig("rewards_plot_latex.png", dpi=600)
        if self.example_name is not None:
            np.save(f"qtable-qsim/qtable-{self.example_name}.npy", self.Q_table, allow_pickle=True)


# function for generating random connected graph
def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G


# Function to run the simulation for a single source-destination pair and modes
def simulate_source_destination(params):
    G, sn, dest, modes, graph_name, eps = params
    results = []

    for mode in modes:
        # Initialize the agent for each mode and run the simulation
        agent = QuantumRoutingAgent(G, start_node=sn, destination=dest,
                                    mode=mode, episodes=eps, epsilon=1,
                                    alpha=0.01, gamma=0.9,
                                    qtable=None, save_training=False)
        agent.run_simulation(steps=10)
        if agent.best_fidelity is not None:
            results.append((mode, agent.best_fidelity, agent.best_rate, sn, dest, len(agent.optimal_path), len(agent.optimal_extended_path)))
        else:
            results.append((mode, None, None, None, None, None, None))

    return results