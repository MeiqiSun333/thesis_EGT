# thesis_EGT

## Game.py
This module defines a Game class for a 2×2 game used in the thesis. Payoffs are parameterized by a pair (U, V) and can be optionally normalized relative to the grand mean. 
Important funcitons:
- getPayoffMatrix() builds the payoff matrix.
- playGame(choiceP0, choiceP1, matrix=None) records a play and returns the payoff pair for the chosen actions.
- update_payoffs(dependence) returns payoffs with their average using a payoff dependence parameter.
- prospect_utility(eta, c, w) computes reference-dependent utility for outcome c relative to recent wealth w.
- getQreChance(...) builds utilities and solves the logit QRE with scipy.optimize.root, returning each player’s equilibrium action probability.

## GameAgent.py
This core module implements an adaptive Mesa agent that plays games on an adaptive network. Each agent holds its own game parameters (U, V), risk aversion, rationality, total wealth and recent wealth. At every step the agent selects a neighbor, draws actions from quantal-response probabilities returned by its Game, updates wealth and last utility, imitates the neighbor’s game (logistic switch based on prospect utilities), and mutates to a random game at rate M/N². It can also let the game drift within bounds (not included in the thesis). The network can rewire: the agent removes a safe edge that preserves connectivity and creates a new tie to a second-order or non-neighbor, with probabilities that depend on wealth distance. This module also include a mean-field function for implementing the null model. The class depends on networkx for the graph, mesa.Agent for scheduling, and the Game class for payoffs, utilities, and QRE strategy probabilities.
Important functions:
- get_rewiring_prob(neighbors, alpha, R, neighbor=False): computes normalized logistic scores for choosing which edge to drop or add, based on wealth differences.
- rewire(alpha, beta, rewiring_p): with probability rewiring_p, removes one existing edge and creates a new edge to a second-order neighbor, otherwise to a random non-neighbor.
- getPlayerStrategyProbs(other_agent): queries the linked Game to compute logit-QRE action probabilities for both players, given their eta, rationality, utility function, recent wealth, and payoff dependence.
- mean_field_play(other_agent): plays one game using QRE draws under mean-field assumption.
- step(): the main update, including optionally drifts (not in this thesis), games playing, updating, rewiring, game learning and mutation.

## GamesModel.py
This module defines the GamesModel, a Mesa model that simulates agents playing games on a dynamic network. It builds the initial graph (Watts–Strogatz, scale-free, or Erdős–Rényi), assigns each agent a game and behavioral parameters, and advances the system with a random-order scheduler and a DataCollector that records network and inequality statistics. The model also includes the null mode that pairs agents randomly each step without network dynamics. 
Important functions:
- __init__(...): sets global options (population size, rewiring parameter, payoff dependence, mutation rate, normalization, period/drift for time variation), and creates the network by the chosen generator. It samples each agent’s (U, V) game via stratified sampling, instantiates GameAgents, and configures a DataCollector for network measures, Gini, unique games, and total social utility.
- stratified_sampling(n_agents, space_range, blocked_area=None): creates a dense grid over the specified (U, V) bounds, optionally removes a blocked rectangle and shuffles.

## Simulate.py
This module implements full simulation runs. It returns two pandas DataFrames: model-level metrics and agent-level metrics. 
Main arguments:
- N, network, rewiring_p, alpha: population size, network type, rewiring probability, homophily intensity.
- rat, risk, dependence, dependence_game, mutation_rate: system rationality level, system risk aversion level, payoff dependence, learning intensity, and mutation rate parameter M, changing the system mutation by the definition of M/N², while N is the population size.
- steps: how many steps per run.
- normalizeGames: range from 0 to 1, representing the normalization intensity, 0 for non-normalize and 1 for normalize. Also can be set as True or False.
- null_model: pair agents randomly without network dynamics.
- fix_eta, fix_rationality: control whether to risk aversion and rationality for the null model.
- period, drift, rationality_sigma: and optional variation parameter not included in this thesis. period controls the time varying normalization intensity, drift controls the random walk intensity of agent's payoff set (U,V), rationality_sigma controls the sigma of rationality distribution, changing the tail shape with the system mean value keeping the same.

## config.py:
Default parameters for the simulations, plus network presets and behavioral parameters to be imported by other modules.

## utils.py
Utility helpers to load the config, build parameter-stamped file paths and create folders, convert stringified lists in DataFrames.

## run_simulations.ipynb


## experiments.ipynb


