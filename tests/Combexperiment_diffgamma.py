#%%
import sys
import os
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)
sys.path.append(project_root)

import src.TS as TS

T = 100000
my_generator = np.random.default_rng(12345)
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



more_child_rngs = my_generator.spawn(14)

instance = TS.CSMABInstance(u, T, m, my_generator, num_exp, diffeps=True)
instance.simulate(rewards, available_arms)
instance.markers = {
                    r'CTS-G ($\gamma = 0.01$)': 'o',
                    r'CTS-B': '8',
                    r'CombUCB': 'x',
                    r'CTS-G ($\gamma = 0.1$)': 'h',
                    r'CTS-G ($\gamma = 0.5$)': 'H',
                    r'CTS-G ($\gamma = 1$)': '*',
                    r'CL-SG ($\gamma = 0.01$)': 'v',
                    r'CL-SG ($\gamma = 0.1$)': 'd',
                    r'CL-SG ($\gamma = 0.5$)': '<',
                    r'CL-SG ($\gamma = 1$)': '>',
                    r'CL-LG ($\gamma = 0.01$)': '^',
                    r'CL-LG ($\gamma = 0.1$)': 's',
                    r'CL-LG ($\gamma = 0.5$)': 'p',
                    r'CL-LG ($\gamma = 1$)': 'P',
                }

instance.algorithms = {
                    r'CTS-G ($\gamma = 0.1$)': TS.CombTS_Basic(m, N, more_child_rngs[0], False, epsi = 0.01),  # TS with beta prior
                    r'CL-SG ($\gamma = 0.1$)': TS.CombTS_Basic(m, N, more_child_rngs[2], False, epsi = 0.1),  # TS with normal prior
                    r'CombUCB': TS.CombTS_Basic(m, N, more_child_rngs[3], False, epsi = 0.5),  # TS with normal prior
                    r'CTS-B': TS.CombTS_Basic(m, N, more_child_rngs[4], False, epsi = 1),  # TS with normal prior (single seed)
                }
instance.plot_regrets(errorbar=True, 
                        filename="./results/CSMAB_regrets_10arms_Bernoulli.pdf", 
                        pltlow_bound = False, legend=True)

instance.algorithms = {
                    r'CTS-G ($\gamma = 0.1$)': TS.CombTS_Basic(m, N, more_child_rngs[0], False, epsi = 0.01),  # TS with beta prior
                    r'CL-SG ($\gamma = 0.1$)': TS.CombTS_Basic(m, N, more_child_rngs[2], False, epsi = 0.1),  # TS with normal prior
                    r'CL-LG ($\gamma = 0.1$)': TS.CombTS_Basic(m, N, more_child_rngs[2], False, epsi = 0.1),  # TS with normal prior
                    r'CombUCB': TS.CombTS_Basic(m, N, more_child_rngs[3], False, epsi = 0.5),  # TS with normal prior
                    r'CTS-B': TS.CombTS_Basic(m, N, more_child_rngs[4], False, epsi = 1),  # TS with normal prior (single seed)
                }
instance.plot_regrets(errorbar=True, 
                        filename="./results/CSMAB_regrets_10arms_Bernoulli_all.pdf", 
                        pltlow_bound = False, legend=True)



instance.algorithms = {
                    r'CTS-G ($\gamma = 0.01$)': TS.CombTS_Basic(m, N, more_child_rngs[0], False, epsi = 0.01),  # TS with beta prior
                    r'CTS-G ($\gamma = 0.1$)': TS.CombTS_Basic(m, N, more_child_rngs[2], False, epsi = 0.1),  # TS with normal prior
                    r'CTS-G ($\gamma = 0.5$)': TS.CombTS_Basic(m, N, more_child_rngs[3], False, epsi = 0.5),  # TS with normal prior
                    r'CTS-G ($\gamma = 1$)': TS.CombTS_Basic(m, N, more_child_rngs[4], False, epsi = 1),  # TS with normal prior (single seed)
                }
instance.plot_regrets(errorbar=True, 
                        filename="./results/CSMAB_regrets_10arms_diffgamma_CTS_G.pdf", 
                        pltlow_bound = False, legend=True)

instance.algorithms = {
                    r'CL-SG ($\gamma = 0.01$)':TS.CombTS_Single(m, N, more_child_rngs[5], False, epsi = 0.01),
                    r'CL-SG ($\gamma = 0.1$)': TS.CombTS_Single(m, N, more_child_rngs[6], False, epsi = 0.1),
                    r'CL-SG ($\gamma = 0.5$)': TS.CombTS_Single(m, N, more_child_rngs[7], False, epsi = 0.5),
                    r'CL-SG ($\gamma = 1$)': TS.CombTS_Single(m, N, more_child_rngs[8], False, epsi = 1),
                }
instance.plot_regrets(errorbar=True, 
                        filename="./results/CSMAB_regrets_10arms_diffgamma_CL_SG.pdf", 
                        pltlow_bound = False, legend=True)

instance.algorithms = {
                    r'CL-LG ($\gamma = 0.01$)':TS.CombTS_Single(m, N, more_child_rngs[9], False, True, epsi = 0.01),
                    r'CL-LG ($\gamma = 0.1$)': TS.CombTS_Single(m, N, more_child_rngs[10], False, True, epsi = 0.1),
                    r'CL-LG ($\gamma = 0.5$)': TS.CombTS_Single(m, N, more_child_rngs[11], False, True, epsi = 0.5),
                    r'CL-LG ($\gamma = 1$)': TS.CombTS_Single(m, N, more_child_rngs[12], False, True, epsi = 1),
                }
instance.plot_regrets(errorbar=True, 
                        filename="./results/CSMAB_regrets_10arms_diffgamma_CL_LG.pdf", 
                        pltlow_bound = False, legend=True)
# instance.plot_average_rewards(errorbar=True,
#                         filename="./results/CSMAB_average_rewards_10arms_Bernoulli.pdf")

instance.algorithms = {
                    r'CTS-G ($\gamma = 0.01$)':TS.CombTS_Single(m, N, more_child_rngs[9], False, True, epsi = 0.01),
                    r'CL-SG ($\gamma = 0.01$)': TS.CombTS_Single(m, N, more_child_rngs[10], False, True, epsi = 0.1),
                    r'CL-LG ($\gamma = 0.01$)': TS.CombTS_Single(m, N, more_child_rngs[11], False, True, epsi = 0.5),
                }
instance.plot_regrets(errorbar=True, 
                        filename="./results/CSMAB_regrets_10arms_diffgamma_all.pdf", 
                        pltlow_bound = False, legend=True)
# %%

