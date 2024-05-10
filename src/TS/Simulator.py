from .KL_UCB import KL_UCB_Plus
from .Lower_bound import Lower_bound
from .CombTS import CombTS_Basic, CombTS_Single, CombTS_Single_Aggr
from .CombUCB import CombUCB


import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as stats

class CSMABInstance:
    def __init__(self, u, T, m, rnd_generator, num_exp, bernoulli=True, algorithms=None, markers=None):
        """
        Initializes the Simulator object.

        Parameters:
        - u (list): List of true means for each arm.
        - T (int): Number of time steps.
        - m (int): Number of arms to select in each round. 
        - rnd_generator (object): Random number generator object.
        - num_exp (int): Number of experiments to run.
        - bernoulli (bool): Flag indicating whether rewards are Bernoulli or Gaussian. Default is True.
        - algorithms (dict): Dictionary of algorithms to use. Default is None.
        - markers (dict): Dictionary of markers for the algorithms. Default is None.
        """
        self.u = u
        self.T = T
        self.rnd_generator = rnd_generator
        self.bernoulli = bernoulli
        self.K = len(u)
        self.num_exp = num_exp
        self.m = m

        # Set parameters for the algorithms
        if algorithms is not None:
            self.algorithms = algorithms
        else:
            if bernoulli:
                more_child_rngs = rnd_generator.spawn(5)
                self.algorithms = {
                    r'CTS-B': CombTS_Basic(m, self.K, more_child_rngs[0], True),  # TS with beta prior
                    r'CTS-G': CombTS_Basic(m, self.K, more_child_rngs[1], False),  # TS with normal prior
                    r'FSGPL': CombTS_Single(m, self.K, more_child_rngs[2], False),  # TS with normal prior (single seed)
                    r'CombUCB': CombUCB(m, self.K, more_child_rngs[4]),  # CombUCB
                }
            else:
                more_child_rngs = rnd_generator.spawn(5)
                self.algorithms = {
                    r'CTS-G': CombTS_Basic(m, self.K, more_child_rngs[1], False),  # TS with normal prior
                    r'FSGPL': CombTS_Single(m, self.K, more_child_rngs[2], False),  # TS with normal prior (single seed)
                    r'CombUCB': CombUCB(m, self.K, more_child_rngs[4]),  # CombUCB
                }

        # Set the markers for the algorithms
        if markers is not None:
            self.markers = markers
        else:
            if bernoulli:
                self.markers = {
                    r'CTS-B': 'o',
                    r'CTS-G': 'd',
                    r'FSGPL': 'x',
                    r'FSAGPL': 'v',
                    r'CombUCB': '<',
                }
            else:
                self.markers = {
                    r'CTS-B': 'o',
                    r'CTS-G': 'd',
                    r'FSGPL': 'x',
                    r'FSAGPL': 'v',
                    r'CombUCB': '<',
                }

        # initialize the average rewards and regrets for each algorithm
        self.average_rewards = {}
        self.regrets = {}

        self.opt_average_rewards = np.zeros([self.num_exp, T])

        for algorithm in self.algorithms:
            self.average_rewards[algorithm] = np.zeros([self.num_exp, T])
            self.regrets[algorithm] = np.zeros([self.num_exp, T])

        self.lower_bound_regrets = self.calculate_lower_bound()

    def calculate_lower_bound(self):
        lower_bound = Lower_bound(self.K, self.u, self.m, self.bernoulli, gap_deps=False)
        regrets = np.zeros(self.T)
        for t in range(self.T):
            regrets[t] = lower_bound.lower_bound(t)
        
        return regrets


    def simulate_one_round(self, rewards, available_arms):
        # Simulate the instance
        average_rewards = {}
        regrets = {}
        opt_ave_rewards = np.zeros(self.T)
        
        for algorithm in self.algorithms:
            average_rewards[algorithm] = np.zeros(self.T)
            regrets[algorithm] = np.zeros(self.T)
            #reset parameters in each algorithm
            self.algorithms[algorithm].reset()

        for t in range(self.T):
            for alg in self.algorithms:
                available_arms_index = np.where(available_arms[:,t] == 1)[0]
                #draw an action and update the stats
                algorithm = self.algorithms[alg]
                action = algorithm.draw_action(available_arms_index)
                reward = rewards[action, t]
                algorithm.update_stats(action, reward)
                
                #update the average rewards and regrets
                average_rewards[alg][t] = sum(reward)

                #calculate the optimal action
                if available_arms_index.size <= self.m:
                    regrets[alg][t] = 0
                    if available_arms_index.size > 0:
                        opt_ave_rewards[t] = sum(self.u[available_arms_index])
                else:
                    ind_in_available_arms_index = np.argpartition(self.u[available_arms_index], -self.m)[-self.m:]
                    ind = available_arms_index[ind_in_available_arms_index]
                    regrets[alg][t] = sum(self.u[ind]) - sum(self.u[action])
                    opt_ave_rewards[t] = sum(self.u[ind])
                
        for algorithm in self.algorithms:
            # use np.cumsum to calculate the cumulative sum of the rewards and regrets
            average_rewards[algorithm] = np.cumsum(average_rewards[algorithm])
            regrets[algorithm] = np.cumsum(regrets[algorithm])
            opt_ave_rewards = np.cumsum(opt_ave_rewards)
            # calculate the time average rewards
            average_rewards[algorithm] = [average_rewards[algorithm][t]/(t+1) for t in range(self.T)]
            opt_ave_rewards = [opt_ave_rewards[t]/(t+1) for t in range(self.T)]
        
        return opt_ave_rewards, average_rewards, regrets
    
    # Implement a function that can simulate the instance multiple times according to self.exp_num, save the result for each round to self.average_rewards and self.regrets
    def simulate(self, rewards, available_arms):
        # generate bernoulli random variables based on rewards
        # rewards = self.rnd_generator.binomial(1, rewards)
        # available_arms = self.rnd_generator.binomial(1, available_arms)     

        for exp in range(self.num_exp):
            if self.bernoulli:
                rnd_rewards = self.rnd_generator.binomial(1, rewards)
            else:
                # generate gaussian random variables based on rewards
                rnd_rewards = self.rnd_generator.normal(rewards, 1)
            rnd_available_arms = self.rnd_generator.binomial(1, available_arms)     
            opt_ave_rewards, average_rewards, regrets = self.simulate_one_round(rnd_rewards, rnd_available_arms)
            self.opt_average_rewards[exp] = opt_ave_rewards
            for algorithm in self.algorithms:
                self.average_rewards[algorithm][exp] = average_rewards[algorithm]
                self.regrets[algorithm][exp] = regrets[algorithm]
        
    
    # implement a function that can plot the average rewards for each algorithm with error bars
    def plot_average_rewards(self, errorbar=False, filename="average_rewards.pdf", fontsize = 20, pltshow = False):
        sns.set_theme()
        sns.set_style("whitegrid")
        plt.figure()
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = fontsize
        #plot self.opt_average_rewards
        plt.plot(np.mean(self.opt_average_rewards, axis = 0), label = 'Optimal', marker = '^', markevery=1000)
        for algorithm in self.algorithms:
            plt.plot(np.mean(self.average_rewards[algorithm], axis = 0), label = algorithm, marker = self.markers[algorithm], markevery=1000)
            if errorbar:
                #calculate 95% confidence interval for the mean
                t_critical = stats.t.ppf(0.975, self.num_exp-1)
                ci = t_critical * np.std(self.average_rewards[algorithm], axis = 0) / np.sqrt(self.num_exp)
                plt.fill_between(range(self.T), np.mean(self.average_rewards[algorithm], axis = 0) - ci, 
                                 np.mean(self.average_rewards[algorithm], axis = 0) + ci, alpha = 0.1)
        plt.legend(ncol=2)
        # Create a ScalarFormatter object
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))

        # Apply the formatter to the y-axis
        plt.gca().yaxis.set_major_formatter(formatter)
        # Apply the formatter to the x-axis
        plt.gca().xaxis.set_major_formatter(formatter)
        #plt.grid(True)
        plt.xlabel('t', fontsize=12)
        plt.ylabel('Time-averaged Rewards', fontsize=12)
        plt.savefig(filename, transparent=True, bbox_inches='tight')
        if pltshow:
            plt.show()

    
    # implement a function that can plot the regrets for each algorithm with error bars, each algorithm with its markers.
    def plot_regrets(self, errorbar=False, filename="regrets.pdf", fontsizes=20, pltshow = False,   pltlow_bound = False, use_sciformat = False, legend = False):
        sns.set_theme()
        sns.set_style("whitegrid")
        plt.figure()
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = fontsizes
        if pltlow_bound:
            plt.plot(self.lower_bound_regrets, label = r'Lower Bound', marker = '^', markevery=int(self.T/10))     
        for algorithm in self.algorithms:
            plt.plot(np.mean(self.regrets[algorithm], axis = 0), label = algorithm,
                     marker = self.markers[algorithm], markevery=int(self.T/10))
            if errorbar:
                #calculate 95% confidence interval for the mean
                t_critical = stats.t.ppf(0.975, self.num_exp-1)
                ci = t_critical * np.std(self.regrets[algorithm], axis = 0) / np.sqrt(self.num_exp)
                plt.fill_between(range(self.T), np.mean(self.regrets[algorithm], axis = 0) - ci, 
                                 np.mean(self.regrets[algorithm], axis = 0) + ci, alpha = 0.1)
        
        if legend:
            #plt.legend(loc='upper left', bbox_to_anchor=(-0.6, 1))
            plt.legend(fontsize=20)
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
        plt.savefig(filename, bbox_inches='tight',pad_inches=0.0)
        if pltshow:
            plt.show()

    # use sns.lineplot to plot the regret for each algorithm with error bars
    def plot_regrets_seaborn(self, filename="regrets.pdf"):
        #sns.lineplot(data = self.lower_bound_regrets, label = 'Lower Bound', marker = '^', markevery=1000)
        sns.set_theme()
        sns.set(font_scale=1.5, rc={'text.usetex' : True})
        # Initialize an empty list to store the data
        rows = []
        # Iterate over the dictionary items
        for algorithm, values in self.regrets.items():
            for exp_num, exp_values in enumerate(values):
                for round_num, value in enumerate(exp_values):
                    # Append a tuple for each value
                    if value == np.inf :
                        print(algorithm, exp_num + 1, round_num + 1, value)
                    rows.append((algorithm, exp_num + 1, round_num + 1, value))
            
            # Append the temporary DataFrame to the main DataFrame
        regret_df = pd.DataFrame(rows, columns=['Algorithm', 'Experiment Number', 't', 'regret'])
        sns.lineplot(regret_df, x='t', y='regret', hue='Algorithm')
        plt.legend()
        #plt.grid(True)
        plt.savefig(filename, transparent=True, bbox_inches='tight', format='pdf')
        plt.show()


class CSMABInstance4lowerbound(CSMABInstance):
    #inherite CSMABInstance
    def __init__(self, u, T, m, rnd_generator, num_exp, bernoulli=True, algorithms=None, markers=None):
        """
        Initializes the Simulator object.

        Parameters:
        - u (list): List of true means for each arm.
        - T (int): Number of time steps.
        - m (int): Number of arms to select in each round. 
        - rnd_generator (object): Random number generator object.
        - num_exp (int): Number of experiments to run.
        - bernoulli (bool): Flag indicating whether rewards are Bernoulli or Gaussian. Default is True.
        - algorithms (dict): Dictionary of algorithms to use. Default is None.
        - markers (dict): Dictionary of markers for the algorithms. Default is None.
        """
        self.u = u
        self.T = T
        self.rnd_generator = rnd_generator
        self.bernoulli = bernoulli
        self.K = len(u)
        self.num_exp = num_exp
        self.m = m

        # Set parameters for the algorithms
        if algorithms is not None:
            self.algorithms = algorithms
        else:
            if bernoulli:
                more_child_rngs = rnd_generator.spawn(5)
                self.algorithms = {
                    r'CTS-G': CombTS_Basic(m, self.K, more_child_rngs[1], False),  # TS with normal prior
                }
            else:
                more_child_rngs = rnd_generator.spawn(5)
                self.algorithms = {
                    r'CTS-G': CombTS_Basic(m, self.K, more_child_rngs[1], False),  # TS with normal prior
                }

        # Set the markers for the algorithms
        if markers is not None:
            self.markers = markers
        else:
            if bernoulli:
                self.markers = {
                    r'CTS': 'o',
                    r'CTS-G': 'd',
                    r'FSGPL': 'x',
                    r'FSAGPL': 'v',
                    r'CombUCB': '<',
                }
            else:
                self.markers = {
                    r'CTS': 'o',
                    r'FGPL': 'd',
                    r'FSGPL': 'x',
                    r'FSAGPL': 'v',
                    r'CombUCB': '<',
                }

        # initialize the average rewards and regrets for each algorithm
        self.average_rewards = {}
        self.regrets = {}

        self.opt_average_rewards = np.zeros([self.num_exp, T])

        for algorithm in self.algorithms:
            self.average_rewards[algorithm] = np.zeros([self.num_exp, T])
            self.regrets[algorithm] = np.zeros([self.num_exp, T])

        self.lower_bound_regrets = self.calculate_lower_bound()

    def plot_regrets(self, errorbar=False, filename="regrets.pdf", fontsizes=20, pltshow = False,   pltlow_bound = False, use_sciformat = False, legend = False):
        sns.set_theme()
        sns.set_style("whitegrid")
        plt.figure()
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = fontsizes
        if pltlow_bound:
            plt.plot(self.lower_bound_regrets, label = r'Lower Bound', marker = '^', markevery=int(self.T/10))     
        for algorithm in self.algorithms:
            plt.plot(np.mean(self.regrets[algorithm], axis = 0), label = algorithm,
                     marker = self.markers[algorithm], markevery=int(self.T/10))
            if errorbar:
                #calculate 95% confidence interval for the mean
                t_critical = stats.t.ppf(0.975, self.num_exp-1)
                ci = t_critical * np.std(self.regrets[algorithm], axis = 0) / np.sqrt(self.num_exp)
                plt.fill_between(range(self.T), np.mean(self.regrets[algorithm], axis = 0) - ci, 
                                 np.mean(self.regrets[algorithm], axis = 0) + ci, alpha = 0.1)
        
        if legend:
            #plt.legend(loc='upper left', bbox_to_anchor=(-0.6, 1))
            plt.legend()
        # Create a ScalarFormatter object
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(use_sciformat)
        formatter.set_powerlimits((-2,2))

        # Apply the formatter to the y-axis
        plt.gca().yaxis.set_major_formatter(formatter)
        # Apply the formatter to the x-axis
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.grid(True)
        plt.xlabel('T')
        plt.ylabel('Regret')
        plt.savefig(filename, bbox_inches='tight')
        if pltshow:
            plt.show()

if __name__ == "__main__":
    # module test program
    my_generator = np.random.default_rng()
    T = 10000

    # Generate mean value for 10 arms, the first with mean value 0.9, and rest with mean value 0.8
    u = np.array([0.9] + [0.8]*9)

    epsi = 1/len(u)

    alpha = 0.9

    instance = MABInstance(u, T, alpha, my_generator, 10)

    instance.plot_regrets(filename="../results/regrets_10arms.pdf")
# %%
