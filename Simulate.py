import pandas as pd
import GamesModel as gm
import utils

import importlib
importlib.reload(gm)
importlib.reload(utils)

params = utils.get_config()

def simulate(N = params.n_agents, 
             rewiring_p = params.rewiring_p, 
             alpha = params.alpha, 
             rat = params.rat,
             network = params.default_network, 
             rounds = params.n_rounds, 
             steps = params.n_steps, 
             netRat = 0.1, 
             alwaysOwn = False, 
             alwaysSafe = False, 
             UV=(True,None,None,False), 
             risk_distribution = "default", 
             utility_function = "isoelastic",
             normalizeGames =  False,
             dependence = 1,
             dependence_game  = 0.5,
             risk = 0,
             mutation_rate = 1,
             null_model = False,
             fix_eta=False,
             fix_rationality=False,
             period=0,
             drift=0,
             rationality_sigma=0.5):

    model_data = pd.DataFrame(columns=['Round'])
    agent_data = pd.DataFrame(columns=['Round'])

    for round in range(rounds):
        print("Round:", round)
        model = gm.GamesModel(
                 N = N, 
                 rewiring_p = rewiring_p, 
                 alpha = alpha, 
                 rat = rat,
                 network = network,
                 alwaysOwn = alwaysOwn, 
                 UV = (True, None, None, False), 
                 risk_distribution = risk_distribution, 
                 utility_function = utility_function,
                 normalize_games = normalizeGames,
                 dependence = dependence,
                 dependence_game = dependence_game,
                 risk = risk,
                 mutation_rate = mutation_rate,
                 null_model = null_model,
                 fix_eta = fix_eta,
                 fix_rationality = fix_rationality,
                 period = period,
                 drift = drift,
                 rationality_sigma = rationality_sigma)
        # Step through the simulation.
        for _ in range(steps):
            model.step()
        agent_data = pd.concat([agent_data, model.datacollector.get_agent_vars_dataframe()])
        agent_data['Round'] = agent_data['Round'].fillna(round)
        model_data = pd.concat([model_data, model.datacollector.get_model_vars_dataframe()])
        model_data['Round'] = model_data['Round'].fillna(round)    

    # Split the MultiIndex into separate columns for agent data
    agent_data.reset_index(inplace=True)
    # Subtract 1 from the first element of each tuple in the "index" column
    agent_data[['Step', 'Players']] = pd.DataFrame(agent_data['index'].to_list(), index=agent_data.index)

    # Reorder the columns with 'steps' and 'players' immediately after the index for agnet data
    index_columns = ['Step', 'Players']
    agent_data = agent_data[index_columns + [col for col in agent_data.columns if col not in index_columns]]

    # Drop the original 'index' column for agent data
    agent_data.drop(columns=['index'], inplace=True)
    
    # Subtract 1 from each value in the "Step" column
    agent_data['Step'] = agent_data['Step'] - 1

    # For network data, reset the index and rename the index column to "step"
    model_data.reset_index(inplace=True)
    model_data.rename(columns={"index": "Step"}, inplace=True)

    return model_data, agent_data
