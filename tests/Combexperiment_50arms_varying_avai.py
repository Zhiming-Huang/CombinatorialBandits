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
N = 50
m = 15
delta = np.sqrt(N*np.log(N)/T)
u = np.array(np.full(m, 0.9).tolist() + np.full(N-m, 0.8).tolist())
#u = [0.1,0.05,0.03,0.02,0.01]
epsi = 1/len(u)
num_exp = 100

rewards = np.tile(u, (T, 1)).T #repeat the mean values for all arms T times

u_aval = np.full(N, 0.5)
available_arms = np.tile(u_aval, (T, 1)).T #repeat the mean values for all arms T times

algorithms = {
    r'BG-CTS': TS.CombTS_BG(m, N, my_generator, 1/2, 1/2),
    r'CTS-B': TS.CombTS_Basic(m, N, my_generator, True),  # TS with beta prior
    r'CTS-G ($\gamma = 0.01$)': TS.CombTS_Basic(m, N, my_generator, False, epsi = 0.01),  # TS with normal prior
    r'CTS-G ($\gamma = 0.1$)': TS.CombTS_Basic(m, N, my_generator, False, epsi = 0.1),  # TS with normal prior
    r'CTS-G ($\gamma = 0.5$)': TS.CombTS_Basic(m, N, my_generator, False, epsi = 0.5),  # TS with normal prior
    r'CTS-G ($\gamma = 1$)': TS.CombTS_Basic(m, N, my_generator, False, epsi = 1),  # TS with normal prior
    r'CL-SG ($\gamma = 0.01$)': TS.CombTS_Single(m, N, my_generator, False, epsi = 0.01),  # TS with normal prior (single seed)
    r'CL-SG ($\gamma = 0.1$)': TS.CombTS_Single(m, N, my_generator, False, epsi = 0.1),  # TS with normal prior (single seed)
    r'CL-SG ($\gamma = 0.5$)': TS.CombTS_Single(m, N, my_generator, False, epsi = 0.5),  # TS with normal prior (single seed)
    r'CL-SG ($\gamma = 1$)': TS.CombTS_Single(m, N, my_generator, False, epsi = 1),  # TS with normal prior (single seed)
    r'CL-LG ($\gamma = 0.01$)': TS.CombTS_Single(m, N, my_generator, False, True, epsi = 0.01),  # TS with normal prior (single seed)
    r'CL-LG ($\gamma = 0.1$)': TS.CombTS_Single(m, N, my_generator, False, True, epsi = 0.1),  # TS with normal prior (single seed)
    r'CL-LG ($\gamma = 0.5$)': TS.CombTS_Single(m, N, my_generator, False, True, epsi = 0.5),  # TS with normal prior (single seed)
    r'CL-LG ($\gamma = 1$)': TS.CombTS_Single(m, N, my_generator, False, True, epsi = 1),  # TS with normal prior (single seed)
    r'CombUCB': TS.CombUCB(m, N, my_generator),  # CombUCB
}


markers = {
    r'CTS-G ($\gamma = 0.1$)': '.',
    r'CL-SG ($\gamma = 0.1$)': ',',
    r'CombUCB': 'o',
    r'BG-CTS': 'v',
    r'CTS-B': '^',
    r'CTS-G ($\gamma = 0.01$)': '<',
    r'CTS-G ($\gamma = 0.5$)': '>',
    r'CTS-G ($\gamma = 1$)': 's',
    r'CL-SG ($\gamma = 0.01$)': '1',
    r'CL-SG ($\gamma = 0.1$)': '2',
    r'CL-SG ($\gamma = 0.5$)': '3',
    r'CL-SG ($\gamma = 1$)': '4',
    r'CL-LG ($\gamma = 0.01$)': 's',
    r'CL-LG ($\gamma = 0.1$)': 'p',
    r'CL-LG ($\gamma = 0.5$)': '*',
    r'CL-LG ($\gamma = 1$)': 'h',
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



# select algorithms with only CTS-G and CL-SG with gamma = 0.1 and CombUCB and BG-CTS
#
algorithms_to_plot_with_legend_name= {
    r'CTS-G ($\gamma = 0.1$)': r'CTS-G',
    r'BG-CTS': r'BG-CTS',
    r'CombUCB': r'CombUCB',
    r'CTS-B': r'CTS-B',
    r'CL-SG ($\gamma = 0.1$)': r'CL-SG',
}

colors = sns.color_palette()
colors = {
    r'BG-CTS': colors[0],
    r'CTS-B': colors[1],
    r'CTS-G ($\gamma = 0.1$)': colors[2],
    r'CL-SG ($\gamma = 0.1$)': colors[3],
    r'CombUCB': colors[4],
}

sns.set_theme()
sns.set_style("whitegrid")
plt.figure(figsize=(4,3))
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 10
for key, value in algorithms_to_plot_with_legend_name.items():
    #downsample the time
    T_sampled = np.arange(0, T, 10)
    plt.plot(T_sampled, np.mean(instance.regrets[key], axis = 0)[T_sampled], label = value,
                marker = markers[key], markevery=1000, color = colors[key], linewidth = 1)
    if errorbar:
        #calculate 95% confidence interval for the mean
        t_critical = stats.t.ppf(0.975, num_exp-1)
        ci = t_critical * np.std(instance.regrets[key], axis = 0) / np.sqrt(num_exp)
        plt.fill_between(T_sampled, np.mean(instance.regrets[key], axis = 0)[T_sampled] - ci[T_sampled], 
                            np.mean(instance.regrets[key], axis = 0)[T_sampled] + ci[T_sampled], alpha = 0.1, color = colors[key])

if legend:
    #plt.legend(loc='upper left', bbox_to_anchor=(-0.6, 1))
    plt.legend(fontsize=10, ncol = 2)
# Create a ScalarFormatter object
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# Apply the formatter to the y-axis
plt.gca().yaxis.set_major_formatter(formatter)
# Apply the formatter to the x-axis
plt.gca().xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.xlabel('t',fontsize=10)
plt.ylabel('Regret',fontsize=10)
plt.yscale('log')
plt.savefig("results/CSMAB_regrets_50arms_Bernoulli_varying_availability.pdf", bbox_inches='tight')
plt.show()

#%%
########################################################
##plot regret of all algorithms
algorithms_to_plot = {
    r'CTS-G ($\gamma = 0.1$)': r'CTS-G',
    r'BG-CTS': r'BG-CTS',
    r'CombUCB': r'CombUCB',
    r'CTS-B': r'CTS-B',
    r'CL-SG ($\gamma = 0.1$)': r'CL-SG',
    r'CL-LG ($\gamma = 0.1$)': r'CL-LG',
}
colors = sns.color_palette()
colors = {
    r'BG-CTS': colors[0],
    r'CTS-B': colors[1],
    r'CTS-G ($\gamma = 0.1$)': colors[2],
    r'CL-SG ($\gamma = 0.1$)': colors[3],
    r'CombUCB': colors[4],
    r'CL-LG ($\gamma = 0.1$)': colors[5],
}
plt.figure(figsize=(4,3))
sns.set_theme()
sns.set_style("whitegrid")
plt.figure(figsize=(4,3))
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 10
for key, value in algorithms_to_plot.items():
    T_sampled = np.arange(0, T, 10)
    plt.plot(T_sampled, np.mean(instance.regrets[key], axis = 0)[T_sampled], label = value,
                marker = markers[key], markevery=1000, color = colors[key], linewidth = 1)
    if errorbar:
        #calculate 95% confidence interval for the mean
        t_critical = stats.t.ppf(0.975, num_exp-1)
        ci = t_critical * np.std(instance.regrets[key], axis = 0) / np.sqrt(num_exp)
        plt.fill_between(T_sampled, np.mean(instance.regrets[key], axis = 0)[T_sampled] - ci[T_sampled], 
                            np.mean(instance.regrets[key], axis = 0)[T_sampled] + ci[T_sampled], alpha = 0.1, color = colors[key])
if legend:
    #plt.legend(loc='upper left', bbox_to_anchor=(-0.6, 1))
    plt.legend(fontsize=10, ncol = 2)
# Create a ScalarFormatter object
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# Apply the formatter to the y-axis
plt.gca().yaxis.set_major_formatter(formatter)
# Apply the formatter to the x-axis
plt.gca().xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.yscale('log')
plt.xlabel('t',fontsize=10)
plt.ylabel('Regret',fontsize=10)
plt.savefig("results/CSMAB_regrets_50arms_Bernoulli_all_varying_availability.pdf", bbox_inches='tight')
plt.show()

# %%
########################################################
##plot regret of CTS-G with different gamma, and CL-SG with different gamma
algorithms_to_plot = [
    r'CTS-G ($\gamma = 0.01$)',
    r'CTS-G ($\gamma = 0.1$)',
    r'CTS-G ($\gamma = 0.5$)',
    r'CTS-G ($\gamma = 1$)',
    r'CL-SG ($\gamma = 0.01$)',
    r'CL-SG ($\gamma = 0.1$)',
    r'CL-SG ($\gamma = 0.5$)',
    r'CL-SG ($\gamma = 1$)',
]

colors = sns.color_palette()
colors = {
    r'CTS-G ($\gamma = 0.01$)': colors[0],
    r'CTS-G ($\gamma = 0.1$)': colors[1],
    r'CTS-G ($\gamma = 0.5$)': colors[2],
    r'CTS-G ($\gamma = 1$)': colors[3],
    r'CL-SG ($\gamma = 0.01$)': colors[4],
    r'CL-SG ($\gamma = 0.1$)': colors[5],
    r'CL-SG ($\gamma = 0.5$)': colors[6],
    r'CL-SG ($\gamma = 1$)': colors[7],
}

plt.figure(figsize=(4,3))
sns.set_theme()
sns.set_style("whitegrid")
plt.figure(figsize=(4,3))
plt.rcParams['text.usetex'] = True
fontsize=10
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
    plt.legend(fontsize=8, ncol = 2)
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
# Apply the formatter to the y-axis
plt.gca().yaxis.set_major_formatter(formatter)
# Apply the formatter to the x-axis
plt.gca().xaxis.set_major_formatter(formatter)
plt.yscale('log')
plt.xlabel('t',fontsize=10)
plt.ylabel('Regret',fontsize=10)
plt.savefig("results/CSMAB_regrets_50arms_Bernoulli_CTS-G_CL-SG_varying_alpha.pdf", bbox_inches='tight')
plt.show()

# %%
########################################################
##plot regret of CTS-G with different gamma, and CL-SG with different gamma, and CL-LG with different gamma
algorithms_to_plot = [
    r'CTS-G ($\gamma = 0.01$)',
    r'CL-SG ($\gamma = 0.1$)',
    r'CL-LG ($\gamma = 0.01$)',
    r'CL-LG ($\gamma = 0.1$)',  
    r'CL-LG ($\gamma = 0.5$)',
    r'CL-LG ($\gamma = 1$)',
]

colors = sns.color_palette()
colors = {
    r'CTS-G ($\gamma = 0.01$)': colors[0],
    r'CL-SG ($\gamma = 0.1$)': colors[1],
    r'CL-LG ($\gamma = 0.01$)': colors[2],
    r'CL-LG ($\gamma = 0.1$)': colors[3],
    r'CL-LG ($\gamma = 0.5$)': colors[4],
    r'CL-LG ($\gamma = 1$)': colors[5],
}

plt.figure(figsize=(4,3))
sns.set_theme()
sns.set_style("whitegrid")
plt.figure(figsize=(4,3))
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 10
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
    plt.legend(fontsize=8, ncol = 2)
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
# Apply the formatter to the y-axis
plt.gca().yaxis.set_major_formatter(formatter)
# Apply the formatter to the x-axis
plt.gca().xaxis.set_major_formatter(formatter)
plt.yscale('log')
plt.xlabel('t',fontsize=10)
plt.ylabel('Regret',fontsize=10)
plt.savefig("results/CSMAB_regrets_50arms_Bernoulli_CTS-G_CL-SG_CL-LG_varying_alpha.pdf", bbox_inches='tight')
plt.show()




# %%
