import random as rand
from random import random
import networkx as nx
import numpy as np
from mesa import Agent
import copy

import Game
import importlib
importlib.reload(Game)

class GameAgent(Agent):

    def __init__(self, id, model, rewiring_p = 0, alpha = 0, beta = 2, rat = 0, uvpay = (0,0), UV = (True, None, None, False), risk_aversion_distribution =  "default", risk = 0, mutation_rate = 1, period=0, drift=0):
    
        super().__init__(id, model)
        self.id = id
        if rat > 0:
            self.rationality = rat
        else: 
            self.rationality = 0
        self.alpha = alpha              
        self.beta = beta              
        self.rewiring_p = rewiring_p
        self.wealth = 0             
        self.payoff_list = [0] * 5
        self.recent_wealth = 0
        self.period = period
        self.drift = drift

        self.mutation_rate = mutation_rate

        self.model = model
        self.posiVals = [15, 6]
        self.games_played = 0

        self.last_action = -1
        self.last_utility = 0

        # Each agent has a risk aversion parameter
        if getattr(model, "null_model", False) or getattr(model, "fix_eta", False):
            self.eta = model.null_eta
        else:
            if risk_aversion_distribution == "uniform":
                self.eta = np.random.rand()*2      
            elif risk_aversion_distribution == "default":
                self.eta = np.random.lognormal(mean=np.log(1.4**2 / np.sqrt(0.5**2 + 1.4**2)), sigma=np.sqrt(np.log(1 + (0.5**2 / 1.4**2)))) + risk

        if getattr(model, "null_model", False) or getattr(model, "fix_rationality", False):
            self.rationality = model.null_rationality
        else:
            self.rationality = rat       
        
        # eta_base is the default risk aversion parameter
        self.eta_base = self.eta        

        # Each agent has a game
        if UV[0]:
            self.game = Game.Game((uvpay[0], uvpay[1]), self.model.normalize_games, model=self.model, period=self.period)
            self.model.games.append(self.game)
        if not UV[0]:
            self.game = Game.Game((UV[1],UV[2]), self.model.normalize_games, model=self.model, period=self.period)
            self.model.games.append(self.game)
    
    def weighted_sum(self, pay_off_list):
        if len(pay_off_list) != 5:
            raise ValueError("Input list must contain exactly 5 numbers")

        weights = np.full(5, 1/5)
        total_weighted_sum = sum(num * weight for num, weight in zip(pay_off_list, weights))
        
        return total_weighted_sum
    
    def fifo_shift_payoff(self, payoff_list, payoff):

        if len(payoff_list) == 0:
            return payoff_list
        for i in range(len(payoff_list) - 1):
            payoff_list[i] = payoff_list[i + 1]
        
        payoff_list[-1] = payoff
        
        return payoff_list
            
    def get_rewiring_prob(self, neighbors, alpha, R, neighbor=False):

        distances = [np.abs(self.wealth - self.model.agents[neighbor].wealth) for neighbor in neighbors]
        dist_array = np.array(distances)

        # Prevent overflow in exponentiation
        limit = 600
        exponent = alpha * (dist_array - R)
        exponent = np.clip(exponent, -limit, limit)

        # Logistic connection probability (SDA form)
        P_con = 1 / (1 + np.exp(exponent))

        if neighbor:
                P_con = 1 - P_con  # Invert: higher distance → higher probability
        if np.sum(P_con) == 0:
            P_con = np.ones_like(P_con) / len(P_con)
        else:
            P_con /= np.sum(P_con)
        return P_con

    
    def get_non_neighbors(self):

        node = self.id
        all_nodes = list(self.model.graph.nodes())
        # Remove node A from the list
        all_nodes.remove(node)
        # Get first-order neighbors of node A
        first_order_neighbors = list(self.model.graph.neighbors(node))
        # Remove first-order neighbors from the list
        non_neighbors = [node for node in all_nodes if node not in first_order_neighbors]
        return non_neighbors

    def get_second_order_neighbors(self):
        node = self.id
        # Get the first-order neighbors of node B
        first_order_neighbors = set(self.model.graph.neighbors(node))
        # Initialize a set to store second-order neighbors
        second_order_neighbors = set()
        # Iterate over the first-order neighbors
        for neighbor in first_order_neighbors:
            # Get the neighbors of the current neighbor excluding node B and its first-order neighbors
            second_order_neighbors.update(set(self.model.graph.neighbors(neighbor)) - first_order_neighbors - {node})
        return list(second_order_neighbors)
    
    def get_valid_neighbors(self):
        valid_neighbors = []
        node = self.id
        # Get all neighbors of the node
        neighbors = list(self.model.graph.neighbors(node))
        # Iterate through each neighbor
        for neighbor in neighbors:
            # Make a copy of the graph
            graph_copy = copy.deepcopy(self.model.graph)
            graph_copy.remove_edge(node, neighbor)
            # Check if removing the connection with 'node' will not disconnect the network
            if nx.is_connected(graph_copy):
                valid_neighbors.append(neighbor)
        return valid_neighbors
    
    def rewire(self, alpha, beta, rewiring_p):
        # Randomly determine if rewiring probability threshold is met
        if np.random.uniform() < rewiring_p:
            # Only rewire edge if it can be done without disconnecting the network
            candidates_removal = self.get_valid_neighbors()
            if len(candidates_removal) > 1:
                self.model.e_n += 1
                # Calculate probabilities of removal
                P_con = self.get_rewiring_prob(candidates_removal, alpha, beta, neighbor=True)
                # Make choice from first-order neighbours based on probability
                removed_neighbor = np.random.choice(candidates_removal, p=P_con)
                self.model.graph.remove_edge(self.id, removed_neighbor)

                # Add an edge if the agent has second order neighbours
                candidates_connection = self.get_second_order_neighbors()
                if len(candidates_connection) > 0:
                    P_con = self.get_rewiring_prob(candidates_connection, alpha, beta, neighbor=False)
                    # Make choice from second-order neighbours based on probability
                    new_neighbor = np.random.choice(candidates_connection, p=P_con)
                    self.model.graph.add_edge(self.id, new_neighbor)

                # Else make a connection with a random node
                else:
                    candidates_connection = self.get_non_neighbors()
                    new_neighbor = np.random.choice(candidates_connection)
                    self.model.graph.add_edge(self.id, new_neighbor)

    def getPlayerStrategyProbs(self, other_agent):

        p0_Prob_S0 , p1_Prob_S0 = self.game.getQreChance(self.rationality, other_agent.rationality, self.eta, other_agent.eta, self.model.utility_function, self.recent_wealth, other_agent.recent_wealth, self.model.dependence)    
        return(p0_Prob_S0 , p1_Prob_S0)  


    def mean_field_play(self, other_agent):

        p0_prob, p1_prob = self.game.getQreChance(
            self.rationality, other_agent.rationality,
            self.eta, other_agent.eta,
            self.model.utility_function,
            self.recent_wealth, other_agent.recent_wealth,
            dependence=self.model.dependence
        )

        strategy0 = 0 if np.random.rand() < p0_prob else 1
        strategy1 = 0 if np.random.rand() < p1_prob else 1

        payoff_A, payoff_B = self.game.adjust_payoffs_by_dependence(
            self.game.getPlayerCells(0),
            other_agent.game.getPlayerCells(0),
            self.model.dependence
        )
        payoff_matrix = [
            [(payoff_A[0], payoff_B[0]), (payoff_A[2], payoff_B[1])],
            [(payoff_A[1], payoff_B[2]), (payoff_A[3], payoff_B[3])]
        ]

        p0_payoff, p1_payoff = self.game.playGame(strategy0, strategy1, matrix=payoff_matrix)

        self.wealth += p0_payoff
        self.payoff_list = self.fifo_shift_payoff(self.payoff_list, p0_payoff)
        self.recent_wealth = self.weighted_sum(self.payoff_list)
        self.games_played += 1

        other_agent.wealth += p1_payoff
        other_agent.payoff_list = other_agent.fifo_shift_payoff(other_agent.payoff_list, p1_payoff)
        other_agent.recent_wealth = other_agent.weighted_sum(other_agent.payoff_list)
        other_agent.games_played += 1


    def step(self):

        if self.drift > 0:
            current_u, current_v = self.game.UV
            random_shifts = (np.random.rand(2) * 2) - 1

            delta_u = self.drift * random_shifts[0]
            delta_v = self.drift * random_shifts[1]
            new_u = current_u + delta_u
            new_v = current_v + delta_v
            new_u = np.clip(new_u, -1, 2)
            new_v = np.clip(new_v, -1, 2)
            self.game = Game.Game(
                UV=(new_u, new_v),
                normalize_games=self.model.normalize_games,
                model=self.model,
                period=self.period
            )
            self.model.games.append(self.game)

        # If the node does not have neighbours, add to random node in network
        if self.model.graph.degree(self.id) == 0:
            new_neighbor = rand.choice(list(self.model.graph.nodes()))  
            self.model.graph.add_edge(self.id, new_neighbor)    
            return

        # A neighbor is chosen to play a game with.
        neighId = self.random.choice(list(self.model.graph.neighbors(self.id)))

        other_agent = self.model.schedule.agents[neighId]

        # Compute strategy for both players
        p0_Prob_S0, p1_Prob_S0 = self.getPlayerStrategyProbs(other_agent)

        if self.games_played < 5:
            selfpayoff = self.game.getPlayerCells(0)
            otherpayoff = other_agent.game.getPlayerCells(0)
        else:
            selfpayoff, otherpayoff = self.game.adjust_payoffs_by_dependence(
                self.game.getPlayerCells(0),
                other_agent.game.getPlayerCells(0),
                self.model.dependence
            )

        selfmatrix = [[(selfpayoff[0], otherpayoff[0]), (selfpayoff[2], otherpayoff[1])],[(selfpayoff[1], otherpayoff[2]), (selfpayoff[3], otherpayoff[3])]]

        # Choose strategy game for both players
        P0_strategy = 0 if random() < p0_Prob_S0 else 1
        P1_strategy = 0 if random() < p1_Prob_S0 else 1

        self.last_action = P0_strategy

        # The game is played.
        (payoff0, payoff1) = self.game.playGame(P0_strategy, P1_strategy, matrix = selfmatrix)
        
        self.last_utility = self.game.prospect_utility(self.eta, payoff0, self.recent_wealth)

        # Only active players get their respective payoffs.
        self.wealth += payoff0
        self.payoff_list = self.fifo_shift_payoff(self.payoff_list, payoff0)
        self.recent_wealth = self.weighted_sum(self.payoff_list)
        #TODO Symmetric payoffs
        other_agent.wealth += payoff1
        other_agent.pay_off_list = other_agent.fifo_shift_payoff(other_agent.payoff_list, payoff1)
        other_agent.recent_wealth = other_agent.weighted_sum(other_agent.payoff_list)

        # Add that they played one game
        self.games_played += 1
        #other_agent.games_played += 1
        self.model.e_p += 1

        mutated = False
        adapted = False

        u_self = self.game.prospect_utility(self.eta,
                                      self.recent_wealth,
                                      self.recent_wealth)
        u_other = self.game.prospect_utility(self.eta,
                                            other_agent.recent_wealth,
                                            self.recent_wealth)
        delta_u = u_other - u_self

        prob_switch = 1.0 / (1.0 + np.exp(-self.rationality * delta_u))

        if np.random.rand() < prob_switch:
            d = self.model.dependence_game
            u_new = (1 - d) * self.game.UV[0] + d * other_agent.game.UV[0]
            v_new = (1 - d) * self.game.UV[1] + d * other_agent.game.UV[1]

            self.game = Game.Game(
                UV=(u_new, v_new),
                normalize_games=self.model.normalize_games,
                model=self.model,
                period=self.period
            )
            self.model.games.append(self.game)
            adapted = True

        #random mutation of game
        # Use M/N^2    
        if rand.random() < self.mutation_rate/(self.model.num_agents)**2:
            mutated = True
            if self.model.NH:
                while True:
                    uvpay = np.random.RandomState().rand(2) * 2
                    if uvpay[0] > 1 and uvpay[1] > 1:
                        self.game = Game.Game((uvpay[0], uvpay[1]), self.model.normalize_games)
                        self.model.games.append(self.game)
                        break
            else:
                uvpay = np.random.RandomState().rand(2) * 2
                self.game = Game.Game((uvpay[0], uvpay[1]), self.model.normalize_games)
                self.model.games.append(self.game)

        if (mutated or adapted):
            self.model.e_g += 1

        if self.games_played > 5:
            self.rewire(self.alpha, self.beta, self.rewiring_p)