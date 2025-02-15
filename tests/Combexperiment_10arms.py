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

u_aval = np.full(N, 1)
available_arms = np.tile(u_aval, (T, 1)).T #repeat the mean values for all arms T times

algorithms = {
    r'BG-CTS': TS.CombTS_BG(m, N, my_generator, 1/2, 1/2),
    r'CTS-B': TS.CombTS_Basic(m, N, my_generator, True),  # TS with beta prior
    r'CTS-G': TS.CombTS_Basic(m, N, my_generator, False),  # TS with normal prior
    r'CL-SG': TS.CombTS_Single(m, N, my_generator, False),  # TS with normal prior (single seed)
    r'CL-LG': TS.CombTS_Single(m, N, my_generator, False, True),  # TS with normal prior (single seed)
    r'CombUCB': TS.CombUCB(m, N, my_generator),  # CombUCB
}


markers = {
    r'BG-CTS': 'o',
    r'CTS-B': 'x',
    r'CTS-G': 's',
    r'CL-SG': 'd',
    r'CL-LG': 'v',
    r'CombUCB': 'p'
}

instance = TS.CSMABInstance(u, T, m, my_generator, num_exp, algorithms = algorithms, markers =   markers)
instance.simulate(rewards, available_arms)
#instance.plot_regrets(errorbar=True, 
#                        filename="./results/CSMAB_regrets_10arms_Bernoulli.pdf", 
#                        pltlow_bound = False, legend=True)




#%%
##plot regret of all algorithms except CL-LG with a predefined color using sns default color palette
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.ticker import ScalarFormatter


errorbar = True
legend = True

colors = sns.color_palette()
colors = {
    r'BG-CTS': colors[0],
    r'CTS-B': colors[1],
    r'CTS-G': colors[2],
    r'CL-SG': colors[3],
    r'CL-LG': colors[4],
    r'CombUCB': colors[5]
}
algorithms_to_plot = [algorithm for algorithm in algorithms if algorithm != r'CL-LG']

sns.set_theme()
sns.set_style("whitegrid")
plt.figure(figsize=(4,3))
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
for algorithm in algorithms_to_plot:
    #downsample the time
    T_sampled = np.arange(0, T, 10)
    plt.plot(T_sampled, np.mean(instance.regrets[algorithm], axis = 0)[T_sampled], label = algorithm,
                marker = markers[algorithm], markevery=1000, color = colors[algorithm], linewidth = 1)
    if errorbar:
        #calculate 95% confidence interval for the mean
        t_critical = stats.t.ppf(0.975, num_exp-1)
        ci = t_critical * np.std(instance.regrets[algorithm], axis = 0) / np.sqrt(num_exp)
        plt.fill_between(T_sampled, np.mean(instance.regrets[algorithm], axis = 0)[T_sampled] - ci[T_sampled], 
                            np.mean(instance.regrets[algorithm], axis = 0)[T_sampled] + ci[T_sampled], alpha = 0.1, color = colors[algorithm])

if legend:
    #plt.legend(loc='upper left', bbox_to_anchor=(-0.6, 1))
    plt.legend(fontsize=10)
# Create a ScalarFormatter object
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# Apply the formatter to the y-axis
plt.gca().yaxis.set_major_formatter(formatter)
# Apply the formatter to the x-axis
plt.gca().xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.xlabel('t',fontsize=20)
plt.ylabel('Regret',fontsize=20)
plt.ylim(0, 2500)
plt.savefig("results/CSMAB_regrets_10arms_Bernoulli.pdf", bbox_inches='tight')
plt.show()

#%%
########################################################
##plot regret of all algorithms
algorithms_to_plot = list(algorithms.keys())
colors = sns.color_palette()
colors = {
    r'BG-CTS': colors[0],
    r'CTS-B': colors[1],
    r'CTS-G': colors[2],
    r'CL-SG': colors[3],
    r'CL-LG': colors[4],
    r'CombUCB': colors[5]
}
plt.figure(figsize=(4,3))
sns.set_theme()
sns.set_style("whitegrid")
plt.figure(figsize=(4,3))
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
for algorithm in algorithms_to_plot:
    T_sampled = np.arange(0, T, 10)
    plt.plot(T_sampled, np.mean(instance.regrets[algorithm], axis = 0)[T_sampled], label = algorithm,
                marker = markers[algorithm], markevery=1000, color = colors[algorithm], linewidth = 1)
    if errorbar:
        #calculate 95% confidence interval for the mean
        t_critical = stats.t.ppf(0.975, num_exp-1)
        ci = t_critical * np.std(instance.regrets[algorithm], axis = 0) / np.sqrt(num_exp)
        plt.fill_between(T_sampled, np.mean(instance.regrets[algorithm], axis = 0)[T_sampled] - ci[T_sampled], 
                            np.mean(instance.regrets[algorithm], axis = 0)[T_sampled] + ci[T_sampled], alpha = 0.1, color = colors[algorithm])
if legend:
    #plt.legend(loc='upper left', bbox_to_anchor=(-0.6, 1))
    plt.legend(fontsize=10)
# Create a ScalarFormatter object
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# Apply the formatter to the y-axis
plt.gca().yaxis.set_major_formatter(formatter)
# Apply the formatter to the x-axis
plt.gca().xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.xlabel('t',fontsize=20)
plt.ylabel('Regret',fontsize=20)
plt.savefig("results/CSMAB_regrets_10arms_Bernoulli_all.pdf", bbox_inches='tight')
plt.show()

# %%

