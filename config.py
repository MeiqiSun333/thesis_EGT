### config.py
# A configuration file with hyperparameters.
###

labelsize = 20
titlesize = 22
subtitlesize = 20
ticksize = 16

n_steps = 480             # Must be a multiple of 8
n_agents = 300
n_rounds = 3

rewiring_p = 0.6          # Has to be between 0 and 1
alpha = 0.5               # alpha is the homophilic parameter
beta = 2                # beta is the homophilic parameter

rat = 0.1


networks = [("HK", 4, 0.3), ("RR", 4, 1), ("WS", 4, 0.2)]
default_network = ("WS", 4, 0.2)
null_network = ("ER", 0.02, 1)




