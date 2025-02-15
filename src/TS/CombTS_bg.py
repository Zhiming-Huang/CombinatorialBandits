#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:45:35 2024

@author: Zhiming Huang
"""

import numpy as np
import matplotlib.pyplot as plt

class CombTS_BG:
    #implement Combinatorial Thompson sampling for Beta distribution or Normal distribution
    def __init__(self, m, K, rnd_generator,sigma, sigma_prior, lamda = 0):
        # initialize the action set
       # self.actions = actions
        self.m = m  
        self.K = K
        # initialize the probabiltiy distribution
        # self.eta = np.log(K)/t
        self.t = 1
        self.sigma = sigma
        self.sigma_prior = sigma_prior
        self.lamda = lamda

        
        self.muhats = np.zeros(self.K)
        self.sigmapost = 10**8 * np.ones(self.K)

        #initialize the random generator
        self.rnd_generator = rnd_generator

    def reset(self):
        #reset all the parameters
        self.t = 1
        self.muhats = np.zeros(self.K)
        self.sigmapost = 10**8 * np.ones(self.K)

    def fbonuses(self, lamda = 0.01):
        if self.t < np.exp(1):
            self.t = np.exp(1)
        if lamda == 0:
            cst = 0
        else:
            cst = np.log(1+np.exp(1)/lamda)
        
        f = (1+lamda)* 2 * (np.log(self.t) + (self.m+2)*np.log(np.log(self.t)) + self.m/2 * cst )

        return f

    def gbonuses(self, lamda = 0.01):
        f = self.fbonuses(lamda)
        if self.t < 2:
            l = 1
        else:
            l = np.log(self.t)
        return f/l   

    def draw_action(self, available_arms):
        # play the first self.m arms with the highest thompson sampling value sampled from normal distribution
        if len(available_arms) <= self.m:
            return available_arms
        estimate_mean = np.zeros(self.K)
            #seed = self.rnd_generator.normal(0, 1, self.K)
        
        estimate_mean = self.rnd_generator.normal(self.muhats, np.sqrt(self.gbonuses(lamda = self.lamda))  * self.sigmapost)
        # for k in range(self.K):
        #     if self.alpha[k] == 0 and self.beta[k] == 0:
        #         estimate_mean[k] = self.rnd_generator.normal(0, np.sqrt(self.epsi*self.m*np.log(self.t))/np.sqrt(self.alpha[k]+self.beta[k]+1))
        #     else:
        #         estimate_mean[k] = self.rnd_generator.normal(self.alpha[k]/(self.alpha[k]+self.beta[k]), np.sqrt(self.epsi*self.m*np.log(self.t)/(self.alpha[k]+self.beta[k]+1)))
        
        
        ind = np.argpartition(estimate_mean[available_arms], -self.m)[-self.m:]
        return available_arms[ind]
            

    def update_stats(self, action, r):
        # update talpha and beta
        self.t += 1
        sigmaprev = self.sigmapost[action].copy()
        self.sigmapost[action] = np.sqrt(1/(1/sigmaprev**2 + 1/self.sigma**2))
        self.muhats[action] = self.sigmapost[action]**2 * (self.muhats[action] / sigmaprev**2 + r / self.sigma_prior**2)

if __name__ == "__main__":
    # module test program
    my_generator = np.random.default_rng()

    instance = CombTS_BG(2,10, my_generator, 1/2, 1/2)

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


    # generate arm availability for each round by drawing a bernoulli random variable
    available_arms = np.zeros((N, T))
    for t in range(T):
        available_arms[:,t] = my_generator.binomial(1,1,[N])
        


    for t in range(T):
        #get the indices for the available arms
        available_arms_index = np.where(available_arms[:,t] == 1)[0]

        action = instance.draw_action(available_arms_index)
        r = rewards[action,t]
        instance.update_stats(action, r)
        reward[t] = np.sum(r)
        
        
    cumureward = np.cumsum(reward)


    for t in range(T):
        time_ave_reward[t] = cumureward[t]/(t+1)


    l1 = plt.plot(time_ave_reward, label = 'CombTS_BG')


    plt.legend()

# %%