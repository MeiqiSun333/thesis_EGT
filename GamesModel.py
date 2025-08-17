from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import inequalipy   
import networkx as nx
import numpy as np

import GameAgent as ga
import importlib
importlib.reload(ga)
import utils

params = utils.get_config()

class GamesModel(Model):

    def __init__(self,
                 N = params.n_agents, 
                 rewiring_p = params.rewiring_p, 
                 alpha = params.alpha, 
                 rat = params.rat,
                 network = params.default_network, 
                 alwaysOwn = False, 
                 UV = (True, None, None, False), 
                 risk_distribution = "uniform", 
                 utility_function = "isoelastic",
                 normalize_games = True,
                 dependence = 1,
                 dependence_game = 0.5,
                 risk = 0,
                 mutation_rate = 1,
                 null_model = False,
                 null_eta=1.4,
                 null_rationality=1.1,
                 fix_eta = False,
                 fix_rationality = False,
                 period = 0,
                 drift=0,
                 rationality_sigma = 0.5):
    
        self.current_step = 0
        
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.netRat = rat
        self.ratFunct = lambda f : f**2
        self.alwaysSafe = alwaysOwn
        self.utility_function = utility_function
        self.NH = UV[3]
        self.running = True
        self.games = [] 
        self.normalize_games = normalize_games
        self.dependence = dependence
        self.dependence_game = dependence_game
        self.risk = risk
        self.mutation_rate = mutation_rate

        self.null_model = null_model
        self.null_eta = null_eta
        self.null_rationality = null_rationality
        self.fix_eta = fix_eta
        self.fix_rationality = fix_rationality

        self.period = period
        self.drift = drift

        # The amount of times the games are updated (i.e the UV space) 
        self.e_g = 0

        # The amount of times the network is updated
        self.e_n = 0

        # The amount of games playes
        self.e_p = 0
        
        # Generate the network.
        if network[0] == 'RR':
            if network[1]%2:
                network[1] -= 1 
            self.graph = nx.random_regular_graph(network[1], N)
        if network[0] == 'WS':
            self.graph = nx.watts_strogatz_graph(N, network[1], network[2])
        if network[0] == 'HK':
            self.graph = nx.powerlaw_cluster_graph(N, int(network[1]/2), network[2])
        if network[0] == 'ER':
            self.graph = nx.erdos_renyi_graph(N, network[1])

        #save mean degree of network
        self.initial_mean_degree = self.get_mean_degree()

        # Create Games using stratified sampling
        if self.NH:
            space = (0, 2, 0, 2)
            blocked_area = (0, 1, 0, 1)
            games = self.stratified_sampling(N, space, blocked_area)
        else:
            space = (-1, 2, -1, 2)  # Assuming UV space ranges from 0 to 2
            games = self.stratified_sampling(N, space)

        # rat_params = np.random.lognormal(mean=0, sigma=0.5, size=len(self.graph.nodes()))
        rationality_mean = np.log(1.1) - (rationality_sigma**2)/2
        rat_params = np.random.lognormal(mean=rationality_mean, sigma=rationality_sigma, size=len(self.graph.nodes()))


        # Create agents.
        self.agents = np.array([])
        for idx, node in enumerate(self.graph):
            if self.null_model or self.fix_rationality:
                rationality = self.null_rationality
            else:
                rationality = rat_params[idx] + self.netRat

            agent = ga.GameAgent(
                node, self,
                rewiring_p=rewiring_p,
                alpha=alpha,
                rat=rationality,
                UV=UV,
                uvpay=games[idx],
                risk_aversion_distribution=risk_distribution,
                risk=self.risk,
                mutation_rate=mutation_rate,
                period=self.period,
                drift=self.drift
            )

            self.agents = np.append(self.agents, agent)
            self.schedule.add(agent)

        def get_uv_for_agent(agent):
            if agent.game is None:
                return None
            return agent.game.UV

        # Collect model timestep data.
        self.datacollector = DataCollector(
            # model_reporters={"M: Mean Degree" : self.get_mean_degree, "M: Var of Degree" : self.get_variance_degree, "M: Avg Clustering" : self.get_clustering_coef, "M: Avg Path Length" : self.get_average_path_length, "Gini Coefficient": self.get_gini_coef,
            #                  "Unique Games": self.get_unique_games, "Degree Distr": self.get_degree_distribution, "e_n": "e_n", "e_g": "e_g", "e_p":"e_p", "Game data": self.get_game_data,
            #                  "Risk Assortativity": self.get_risk_assortativity, "Rationality Assortativity": self.get_rationality_assortativity, "Total Social Utility": self.get_total_social_utility},
            model_reporters={"M: Mean Degree" : self.get_mean_degree, "M: Var of Degree" : self.get_variance_degree, "M: Avg Clustering" : self.get_clustering_coef, "M: Avg Path Length" : self.get_average_path_length, "Gini Coefficient": self.get_gini_coef,
                             "Unique Games": self.get_unique_games, "Total Social Utility": self.get_total_social_utility},
            agent_reporters={"Wealth": "wealth","Player Risk Aversion": "eta", "UV": get_uv_for_agent, "Games played": "games_played", "Recent Wealth": "recent_wealth", "Rationality": "rationality", "Action": "last_action"}
        )


    def stratified_sampling(self, n_agents, space_range, blocked_area=None):
        x_min, x_max, y_min, y_max = space_range

        # Increase the number of samples along each dimension
        num_samples = int(np.sqrt(n_agents)) * 3

        # Generate stratified samples for x and y coordinates
        x_samples = np.linspace(x_min, x_max, num_samples)
        y_samples = np.linspace(y_min, y_max, num_samples)

        # Generate all possible combinations of x and y coordinates
        xy_combinations = [(x, y) for x in x_samples for y in y_samples]

        # Remove samples in the blocked area if specified
        if blocked_area:
            x_blocked_min, x_blocked_max, y_blocked_min, y_blocked_max = blocked_area
            xy_combinations = [xy for xy in xy_combinations if not (x_blocked_min <= xy[0] < x_blocked_max and
                                                                    y_blocked_min <= xy[1] < y_blocked_max)]

        # Randomly shuffle the list of combinations
        np.random.shuffle(xy_combinations)

        # Select the first n_agents combinations as the sampled games
        sampled_uv = xy_combinations[:n_agents]

        return sampled_uv
    
    def get_mean_degree(self):
        total_degree = sum([x[1] for x in self.graph.degree()])
        return (total_degree / self.graph.number_of_nodes())
    
    
    def get_degree_distribution(self):
        return [x[1] for x in self.graph.degree()]
    
    
    def get_variance_degree(self):
        degree_list = [x[1] for x in self.graph.degree()]
        mean = self.get_mean_degree()
        return sum((i - mean) ** 2 for i in degree_list) / len(degree_list)
    
    def get_clustering_coef(self):
        return nx.average_clustering(self.graph)
    
    def get_average_path_length(self):
        return nx.average_shortest_path_length(self.graph)
    
    def get_gini_coef(self):
        wealth = np.array([agent.wealth for agent in self.agents])
        return inequalipy.gini(wealth)


    def get_unique_games(self):
        return list(set([agent.game.UV for agent in self.agents]))
    
    def get_ratio_updating_speed(self):
        if self.e_n == 0 or  self.e_g == 0:
            return 0
        return self.e_g / self.e_n
    
    def get_game_data(self):
        game_data = []
        for game in self.games:
            game_data.append([game.name, game.play_count, game.total_payoff, game.UV])
        return game_data

    def get_risk_assortativity(self):
        G = self.graph

        for agent in self.schedule.agents:
            G.nodes[agent.unique_id]['risk_aversion'] = agent.eta

        values = [agent.eta for agent in self.schedule.agents]
        if np.std(values) < 1e-6:
            return 0.0

        assortativity = nx.numeric_assortativity_coefficient(G, 'risk_aversion')
        return 0.0 if np.isnan(assortativity) else assortativity


    def get_rationality_assortativity(self):
        G = self.graph

        for agent in self.schedule.agents:
            G.nodes[agent.unique_id]['rationality'] = agent.rationality

        values = [agent.eta for agent in self.schedule.agents]
        if np.std(values) < 1e-6:
            return 0.0

        assortativity = nx.numeric_assortativity_coefficient(G, 'rationality')
        return 0.0 if np.isnan(assortativity) else assortativity
    
    def get_total_social_utility(self):
        return sum([agent.last_utility for agent in self.schedule.agents])

    
    def step(self):
        if self.null_model:
            shuffled = np.random.permutation(self.schedule.agents)
            for i in range(0, len(shuffled), 2):
                a1, a2 = shuffled[i], shuffled[i + 1]
                a1.mean_field_play(a2)
                self.e_p += 1

            self.schedule.time += 1
            self.schedule.steps += 1
            self.datacollector.collect(self)
            self.current_step += 1
        else:
            self.schedule.step()
            self.datacollector.collect(self)
            self.current_step += 1

