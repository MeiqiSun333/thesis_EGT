# thesis_EGT

This repository contains the full code for my thesis on evolutionary games played on adaptive networks. The core model is built with Mesa and NetworkX, agents update strategies via logit-QRE and interact on graphs that can rewire over time. The codebase includes: (i) simulation modules (Game.py, GameAgent.py, GamesModel.py, and Simulate.py), (ii) utilities and default configurations (utils.py and config.py), (iii) local and HPC workflows for GSA and Pareto-front exploration (GSA folder and pareto folder), and (iv) analysis notebooks for the Experiments Section. All outputs are written as CSVs for reproducible plotting and statistical analysis.

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

## GSA
This folder contains two scripts for running global sensitivity analysis locally in parallel.

### run_gsa.py
- runs a Morris method using SALib, executes many independent simulations in parallel.
- Command-line arguments: --steps, --repetitions, --out_dir (default Data/GSA), --seed_params (design reproducibility), --seed_sim (simulation RNG base), --workers (0 = auto = CPU−1; 1 = sequential), and --overwrite. The parameter space covers seven inputs with the following bounds: rewiring_p (0.1–0.9), alpha (0–1), rat (0–2), risk (0–2), normalize (0–1), dependence (0–1), and dependence_game (0–1).
- Example usage: python3 run_psa.py --repetitions 10 --steps 200

### run_sobol_gsa.py
- runs a Sobol method using SALib, executes many independent simulations in parallel.
Command-line arguments: --steps, --repetitions, --out_dir (default sobol_GSA), --seed_params, --seed_sim, --n_base (base N for Saltelli), --workers (0 = auto; 1 = sequential), and --overwrite. It varies four inputs with bounds alpha (0–1), rat (0–2), normalize (0–1), and dependence_game (0–1), while holding rewiring_p=0.5, risk=1.0, and dependence=0.0 fixed.
- Example usage: python3 run_sobol_gsa.py --repetitions 5 --steps 200 --n_base 256

## pareto
This folder contains the .py script used for the Pareto analysis, a SLURM submission .sh file to run it on HPC, and a Jupyter .ipynb notebook for analysis.

### run_pareto.py
HPC task script that explores a 7-dimensional parameter space with Latin Hypercube Sampling. Draws 1,024 samples each time over rewiring_p, alpha, rat, risk, normalizeGames, dependence, and dependence_game. Saves model_data_<params>_rep<rep>.csv if not already present.

### run_pareto.sh
SLURM submission script that runs the Python task as an array job, sets up the environment, copies sources to the node’s $TMPDIR, executes run_pareto.py for each SLURM_ARRAY_TASK_ID, and logs outputs. Also includes a cleanup-and-rescue function that automatically rsyncs any results from $TMPDIR back to the final results directory on exit.

### Pareto_Front_Analysis.ipynb
This notebook reads the raw model-level CSVs and concatenates them into a single DataFrame. It reports the minimized Gini Coefficient and maximized Total Social Utility. It then computes the non-dominated set (Pareto frontier) via a dominance check, labels frontier and dominated points, generates figures and small summary tables and writes them back for reuse.

## run_simulations.ipynb
A jupyter notebook that runs simulations for the thesis. It repeatedly calls Simulate.simulate(...) with different settings (network type: HK/ER/WS, rewiring probability, homophily alpha, risk aversion eta, rationality lambda, normalization on/off, dependence and dependence_game, mutation rate), and saves the resulting agent-level and model-level CSVs into labeled Data/... folders for later analysis.

## experiments.ipynb
This is the primary experiment notebook, contains the code that produces the results in the Experiments section of the thesis. It loads simulation outputs, groups them by step and run, and builds key metrics such as the Gini coefficient, recent wealth, network statistics (e.g., clustering, path length, and risk/rationality assortativity) and properties (e.g., risk aversion, rationality, mutation, normalization). The notebook contains experiments to study how rewiring probability and homophily affect outcomes, examine network convergence across HK/WS/ER graphs, the influence of risk aversion, rationality, mutation, and normalization, draw the figure and analysis of two GSA methods. It analyzes wealth distributions using CCDF plots and simple power-law fits, applies basic statistical tests (e.g., t-tests/ANOVA) to quantify differences. The notebook also compares the default model with the null model. It also assembles the Pareto scatter used to describe the inequality–utility trade-off.

## Closing summary
In sum, this project provides a complete, reproducible pipeline: configure defaults (config.py, utils.py), run controlled simulations, scale up to large designs locally (run_gsa.py, run_gsa_sobol.py) or on HPC for Pareto studies (run_pareto.py with the SLURM .sh), and finalize the results in the main experiment and analysis notebooks (experiments.ipynb, run_simulations.ipynb). The repository is organized so that each stage can be executed independently, but all stages share a common data format to make comparison and figure generation straightforward. This structure is intended to support transparent methods, easy replication, and direct reuse of the simulations in the thesis.
