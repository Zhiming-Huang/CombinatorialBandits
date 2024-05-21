#%%
import sys
import os
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)
sys.path.append(project_root)

import src.TS as TS

T = 1000000
my_generator = np.random.default_rng()
'''
Generate mean value for 10 arms, the first 1 arm with mean value 0.9
and rest with mean value uniformly generated from 0.8 to 0.85
'''
N = 50
m = 15
delta = np.sqrt(N*np.log(N)/T)
u = np.array(my_generator.uniform(0.725, 0.75, m).tolist() + my_generator.uniform(0.7, 0.725, N-m).tolist())
#u = [0.1,0.05,0.03,0.02,0.01]
epsi = 1/len(u)
num_exp = 100

rewards = np.tile(u, (T, 1)).T #repeat the mean values for all arms T times

# generate bernoulli random variables based on rewards
rewards = my_generator.binomial(1, rewards)

# generate a list of arm availability for each round by drawing a bernoulli random variable
u_aval = np.full(N, 0.5)
available_arms = np.tile(u_aval, (T, 1)).T #repeat the mean values for all arms T times



instance = TS.CSMABInstance(u, T, m, my_generator, num_exp)
instance.simulate(rewards, available_arms)
instance.plot_regrets(errorbar=True, 
                        filename="./results/CSMAB_regrets_20arms_Bernoulli.pdf", 
                        pltlow_bound = False, legend=True)
# instance.plot_average_rewards(errorbar=True,
#                         filename="./results/CSMAB_average_rewards_20arms_Bernoulli.pdf")

# %%

