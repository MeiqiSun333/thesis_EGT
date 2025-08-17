import math
import numpy as np
from itertools import count
from scipy.optimize import root

class Game:
    _ids = count(1)
    def __init__(self, UV = (3, 5), normalize_games = True, model=None, period=0):
        
        self.name = f"Game_{next(self._ids)}"
        self.UV = UV
        self.UV = tuple(round(x, 1) for x in UV)
        self.play_count = 0
        self.total_payoff = 0
        self.normalize = normalize_games
        self.model = model
        self.period = period

    def getPayoffMatrix(self):
        payoff_matrix = [[(1, 1), (self.UV[0], self.UV[1])], [(self.UV[1], self.UV[0]), (0,0)]]
        
        payoff_matrix = [
            [(1, 1),(self.UV[0], self.UV[1])],
            [(self.UV[1], self.UV[0]), (0, 0)]
        ]

        if self.period and self.period > 0:
            t = self.model.current_step 
            lam = 0.5 + 0.5 * math.sin(2 * math.pi * t / self.period)
        else:
            norm_flag = self.normalize
            if isinstance(norm_flag, bool):
                if not norm_flag:
                    return payoff_matrix
                lam = 1.0
            elif isinstance(norm_flag, (int, float)):
                lam = float(norm_flag)
                lam = max(0.0, min(1.0, lam))
                if lam == 0:
                    return payoff_matrix


        all_payoffs = [p for row in payoff_matrix for pair in row for p in pair]
        mu = sum(all_payoffs) / len(all_payoffs)

        normalized = [
            [(p1 - mu * lam, p2 - mu * lam) for (p1, p2) in row]
            for row in payoff_matrix
        ]
        return normalized

    
    def playGame(self, choiceP0, choiceP1, matrix = None):
        if matrix is not None:
            payoffs = matrix
        else:
            payoffs = self.getPayoffMatrix()

        self.play_count += 1
        payoffs = payoffs[choiceP0][choiceP1]
        self.total_payoff = self.total_payoff + payoffs[0]
        self.total_payoff = self.total_payoff + payoffs[1]

        return payoffs
    
    @staticmethod
    def adjust_payoffs_by_dependence(player_A_payoffs, player_B_payoffs, dependence):
        def adjust_cell(payoff_A, payoff_B):
            avg = (payoff_A + payoff_B) / 2
            new_A = (1 - dependence) * payoff_A + dependence * avg
            new_B = (1 - dependence) * payoff_B + dependence * avg
            return new_A, new_B
        
        pA_c11, pB_c11 = adjust_cell(player_A_payoffs[0], player_B_payoffs[0])
        pA_c21, pB_c21 = adjust_cell(player_A_payoffs[1], player_B_payoffs[1])
        pA_c12, pB_c12 = adjust_cell(player_A_payoffs[2], player_B_payoffs[2])
        pA_c22, pB_c22 = adjust_cell(player_A_payoffs[3], player_B_payoffs[3])
        
        return (pA_c11, pA_c21, pA_c12, pA_c22), (pB_c11, pB_c21, pB_c12, pB_c22)

    
    def update_payoffs(self, dependence = 0):
        player_A_payoffs = self.getPlayerCells(0)  # Returns a tuple of 4 values
        player_B_payoffs = self.getPlayerCells(1)  # Returns a tuple of 4 values

        if self.play_count < 5:
            return player_A_payoffs, player_B_payoffs
        else:     
            adjusted_A, adjusted_B = self.adjust_payoffs_by_dependence(
                player_A_payoffs,
                player_B_payoffs,
                dependence
            )      
            return adjusted_A, adjusted_B

    def getPlayerCells(self, player):
        c11 = self.getPayoffMatrix()[0][0][player]
        c21 = self.getPayoffMatrix()[0][1][player]
        c12 = self.getPayoffMatrix()[1][0][player]
        c22 = self.getPayoffMatrix()[1][1][player]
        return (c11, c21, c12, c22)


    def equations(self, vars, pA_u11, pA_u21, pA_u12, pA_u22, pB_u11, pB_u21, pB_u12, pB_u22, lambA, lambB):
        pc1, pc2 = vars
        
        # Clip probabilities to avoid boundary issues
        pc1 = np.clip(pc1, 1e-10, 1-1e-10)
        pc2 = np.clip(pc2, 1e-10, 1-1e-10)
        
        # Expected utilities for each action
        uA1 = pc2 * pA_u11 + (1 - pc2) * pA_u12
        uA2 = pc2 * pA_u21 + (1 - pc2) * pA_u22
        
        uB1 = pc1 * pB_u11 + (1 - pc1) * pB_u21
        uB2 = pc1 * pB_u12 + (1 - pc1) * pB_u22
        
        # Use log-sum-exp trick for numerical stability
        max_uA = max(lambA * uA1, lambA * uA2)
        max_uB = max(lambB * uB1, lambB * uB2)
        
        # Compute the logit probabilities with numerical stability
        logit_pc1 = np.exp(lambA * uA1 - max_uA) / (np.exp(lambA * uA1 - max_uA) + np.exp(lambA * uA2 - max_uA))
        logit_pc2 = np.exp(lambB * uB1 - max_uB) / (np.exp(lambB * uB1 - max_uB) + np.exp(lambB * uB2 - max_uB))
        
        # Return the difference between current and logit probabilities
        return [logit_pc1 - pc1, logit_pc2 - pc2]

    def proportional_scaling_with_range(self, numbers):
        min_val = min(numbers)
        max_val = max(numbers)
        range_val = max_val - min_val
        
        if range_val == 0:  # All values are the same
            return [0.5 for _ in numbers]  # Return middle values
        
        scaled_values = [(x - min_val) / range_val for x in numbers]
        return scaled_values

    def getQreChance(self, rationalityA, rationalityB, etaA, etaB, utility_function, recent_wealthA, recent_wealthB, dependence=0):
    
        if utility_function == 'prospect':
            uA = lambda x: self.prospect_utility(etaA, x, recent_wealthA)
            uB = lambda x: self.prospect_utility(etaB, x, recent_wealthB)
        else:
            raise ValueError("Invalid utility function specified")

        adjusted_A, adjusted_B = self.update_payoffs(dependence)
        pA_c11, pA_c21, pA_c12, pA_c22 = adjusted_A
        pB_c11, pB_c21, pB_c12, pB_c22 = adjusted_B

        # Calculate utilities
        pA_u11 = uA(pA_c11)
        pA_u21 = uA(pA_c21)
        pA_u12 = uA(pA_c12)
        pA_u22 = uA(pA_c22)

        pB_u11 = uB(pB_c11)
        pB_u21 = uB(pB_c21)
        pB_u12 = uB(pB_c12)
        pB_u22 = uB(pB_c22)

        # Scale utilities to prevent overflow issues
        utils = [pA_u11, pA_u21, pA_u12, pA_u22, pB_u11, pB_u21, pB_u12, pB_u22]
        if any(abs(u) > 100 for u in utils):
            # Apply proportional scaling if utilities are large
            utils = self.proportional_scaling_with_range(utils)
            pA_u11, pA_u21, pA_u12, pA_u22, pB_u11, pB_u21, pB_u12, pB_u22 = utils
        
        lambA, lambB = rationalityA, rationalityB
        
        # Use different initial guesses to improve convergence
        initial_guesses = [(0.5, 0.5), (0.2, 0.8), (0.8, 0.2), (0.2, 0.2), (0.8, 0.8)]
        
        for guess in initial_guesses:
            try:
                # Use the more robust root-finding method
                result = root(
                    self.equations, 
                    guess,
                    args=(pA_u11, pA_u21, pA_u12, pA_u22, pB_u11, pB_u21, pB_u12, pB_u22, lambA, lambB),
                    method='hybr',  # Hybrid method tends to be more robust
                    options={'xtol': 1e-4}  # Slightly relaxed tolerance
                )
                
                if result.success:
                    x, y = result.x
                    # Clip results to valid probability range
                    x = np.clip(x, 0, 1)
                    y = np.clip(y, 0, 1)
                    return x, y
            except Exception as e:
                #print waring
                print(f"Attempt with initial guess {guess} failed with error: {e}")
                continue

        # Try best-response dynamics as a fallback
        try:
            # Compute deterministic best responses
            br_A = 1.0 if (pA_u21 > pA_u11) else 0.0
            br_B = 1.0 if (pB_u12 > pB_u11) else 0.0
            return br_A, br_B
        except:
            # Last resort, return uniform random as in original
            print("All QRE solution methods failed, returning uniform random probabilities")
            return np.random.uniform(0, 1), np.random.uniform(0, 1)
         
    
    def prospect_utility(self, eta, c, w):
        alpha = max(0.2, 1 / (1 + eta))
        beta = max(0.2, 1 / (1 + 0.5 * eta))
        lambd = 1 + 2 * np.tanh(eta)

        delta = c - w
        if delta >= 0:
            return delta ** alpha
        else:
            return -lambd * ((-delta) ** beta)