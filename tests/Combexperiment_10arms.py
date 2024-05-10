#%%
import sys
import os
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)
sys.path.append(project_root)

import src.TS as TS

T = 100000
my_generator = np.random.default_rng()
'''
Generate mean value for 10 arms, the first 1 arm with mean value 0.9
and rest with mean value uniformly generated from 0.8 to 0.85
'''
N = 10
m = 3
delta = np.sqrt(N*np.log(N)/T)
u = np.array(np.full(m, 0.9).tolist() + np.full(N-m, 0.8).tolist())
#u = [0.1,0.05,0.03,0.02,0.01]
epsi = 1/len(u)
num_exp = 100

rewards = np.tile(u, (T, 1)).T #repeat the mean values for all arms T times

u_aval = np.full(N, 0.5)
available_arms = np.tile(u_aval, (T, 1)).T #repeat the mean values for all arms T times




instance = TS.CSMABInstance(u, T, m, my_generator, num_exp)
instance.simulate(rewards, available_arms)
instance.plot_regrets(errorbar=True, 
                        filename="./results/CSMAB_regrets_10arms_Bernoulli.pdf", 
                        pltlow_bound = False, legend=True)

# instance.plot_average_rewards(errorbar=True,
#                         filename="./results/CSMAB_average_rewards_10arms_Bernoulli.pdf")
# %%

