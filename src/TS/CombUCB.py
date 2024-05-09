#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:45:35 2024

@author: Zhiming Huang
"""

import numpy as np
import matplotlib.pyplot as plt

class CombUCB:
    #implement Thompson sampling for beta distribution
    def __init__(self, m, K, rnd_generator):
        # initialize the action set
       # self.actions = actions
        self.m = m  
        self.K = K
        # initialize the probabiltiy distribution
        # self.eta = np.log(K)/t
        self.t = 1
        
        #initialize the parameters for beta priors
        self.alpha = np.zeros(K)
        self.beta = np.zeros(K)

        #initialize the random generator
        self.rnd_generator = rnd_generator

    def reset(self):
        #reset all the parameters
        self.t = 1
        self.alpha = np.zeros(self.K)
        self.beta = np.zeros(self.K)

    def draw_action(self, available_arms):
        if len(available_arms) <= self.m:
            return available_arms        
        # play the first self.m arms with the highest thompson sampling value sampled from normal distribution
        seed = self.rnd_generator.normal(0, 1, self.K)
        estimate_mean = np.zeros(self.K)
        for k in range(self.K):
            if self.alpha[k] == 0 and self.beta[k] == 0:
                estimate_mean[k] = np.inf
            else:
                #calculate ucb value
                estimate_mean[k] = self.alpha[k]/(self.alpha[k]+self.beta[k]) + np.sqrt(1.5*np.log(self.t)/(self.alpha[k]+self.beta[k]))
        ind = np.argpartition(estimate_mean[available_arms], -self.m)[-self.m:]
        return available_arms[ind]



            

    def update_stats(self, action, r):
        # update talpha and beta
        self.alpha[action] += r
        self.beta[action] += 1 - r
        self.t += 1




if __name__ == "__main__":
    # module test program
    my_generator = np.random.default_rng()

    instance = CombUCB(2,10, my_generator)

    T = 1000
    N = 10
    
    #generate a multivariable gaussian distribution  
    lowarm = my_generator.uniform(0.7,0.8,[8, T])
    higharm = my_generator.uniform(0.8,1.0,[2, T])

    #generate bernoulli random variables based on lowarm and higharm
    lowarm = my_generator.binomial(1,lowarm)
    higharm = my_generator.binomial(1,higharm)

    rewards = np.vstack([higharm, lowarm])
    
    reward = np.zeros(T)

    time_ave_reward = np.zeros(T)
    available_arms = np.zeros((N, T))
    for t in range(T):
        available_arms[:,t] = my_generator.binomial(1,0.5,[N])
    for t in range(T):
        available_arms_index = np.where(available_arms[:,t] == 1)[0]
        action = instance.draw_action(available_arms_index)
        r = rewards[action,t]
        instance.update_stats(action, r)
        reward[t] = np.sum(r)
        
    cumureward = np.cumsum(reward)


    for t in range(T):
        time_ave_reward[t] = cumureward[t]/(t+1)


    l1 = plt.plot(time_ave_reward, label = 'CombUCB')


    plt.legend()

# %%