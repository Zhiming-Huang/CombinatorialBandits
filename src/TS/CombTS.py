#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:45:35 2024

@author: Zhiming Huang
"""

import numpy as np
import matplotlib.pyplot as plt

class CombTS_Basic:
    #implement Combinatorial Thompson sampling for Beta distribution or Normal distribution
    def __init__(self, m, K, rnd_generator, bernoulli = True,epsi = 0.1):
        # initialize the action set
       # self.actions = actions
        self.m = m  
        self.K = K
        # initialize the probabiltiy distribution
        # self.eta = np.log(K)/t
        self.t = 1
        self.epsi = epsi
        
        #initialize the parameters for beta priors
        self.alpha = np.zeros(K)
        self.beta = np.zeros(K)

        #initialize the random generator
        self.rnd_generator = rnd_generator
        self.bernoulli = bernoulli

    def reset(self):
        #reset all the parameters
        self.t = 1
        self.alpha = np.zeros(self.K)
        self.beta = np.zeros(self.K)



    def draw_action(self, available_arms):
        # play the first self.m arms with the highest thompson sampling value sampled from normal distribution
        if len(available_arms) <= self.m:
            return available_arms
        estimate_mean = np.zeros(self.K)
        if self.bernoulli:
            for k in range(self.K):
                estimate_mean[k] = self.rnd_generator.beta(self.alpha[k]+1, self.beta[k]+1)

        else:
            #seed = self.rnd_generator.normal(0, 1, self.K)
            for k in range(self.K):
                if self.alpha[k] == 0 and self.beta[k] == 0:
                    estimate_mean[k] = self.rnd_generator.normal(0, np.sqrt(self.epsi*self.m*np.log(self.t))/np.sqrt(self.alpha[k]+self.beta[k]+1))
                else:
                    estimate_mean[k] = self.rnd_generator.normal(self.alpha[k]/(self.alpha[k]+self.beta[k]), np.sqrt(self.epsi*self.m*np.log(self.t)/(self.alpha[k]+self.beta[k]+1)))
        
        ind = np.argpartition(estimate_mean[available_arms], -self.m)[-self.m:]
        return available_arms[ind]
            

    def update_stats(self, action, r):
        # update talpha and beta
        self.alpha[action] += r
        self.beta[action] += 1 - r
        self.t += 1



class CombTS_Single(CombTS_Basic):
    #implement Thompson sampling for beta distribution
    def __init__(self, m, K, rnd_generator, bernoulli = True, least_gaussian= False, epsi = 0.1):
        super().__init__(m, K, rnd_generator, bernoulli)
        self.epsi = epsi
        self.seed = self.rnd_generator.normal(0, 1)
        self.least_gaussian = least_gaussian

    def draw_action(self, available_arms):
        # play the first self.m arms with the highest thompson sampling value sampled from normal distribution
        if len(available_arms) <= self.m:
            return available_arms
        estimate_mean = np.zeros(self.K)
        if self.bernoulli:
            for k in range(self.K):
                estimate_mean[k] = self.rnd_generator.beta(self.alpha[k]+1, self.beta[k]+1)

        else:
            if not self.least_gaussian:
                self.seed = self.rnd_generator.normal(0, 1)
            for k in range(self.K):
                if self.alpha[k] == 0 and self.beta[k] == 0:
                    estimate_mean[k] = self.seed * np.sqrt(self.epsi*np.log(self.t))/np.sqrt(self.alpha[k]+self.beta[k]+1)
                else:
                    estimate_mean[k] = self.alpha[k]/(self.alpha[k]+self.beta[k]) +  self.seed * np.sqrt(self.epsi*np.log(self.t)/(self.alpha[k]+self.beta[k]+1))
        ind = np.argpartition(estimate_mean[available_arms], -self.m)[-self.m:]
        return available_arms[ind]

if __name__ == "__main__":
    # module test program
    my_generator = np.random.default_rng()

    instance = CombTS_Basic(2,10, my_generator, bernoulli = True)
    instance2 = CombTS_Basic(2,10, my_generator, bernoulli = False)
    instance3 = CombTS_Single(2,10, my_generator, bernoulli = False)
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
    reward2 = np.zeros(T)
    reward3 = np.zeros(T)
    time_ave_reward = np.zeros(T)
    time_ave_reward2 = np.zeros(T)
    time_ave_reward3 = np.zeros(T)

    # generate arm availability for each round by drawing a bernoulli random variable
    available_arms = np.zeros((N, T))
    for t in range(T):
        available_arms[:,t] = my_generator.binomial(1,0.5,[N])
        


    for t in range(T):
        #get the indices for the available arms
        available_arms_index = np.where(available_arms[:,t] == 1)[0]

        action = instance.draw_action(available_arms_index)
        r = rewards[action,t]
        instance.update_stats(action, r)
        reward[t] = np.sum(r)

        action2 = instance2.draw_action(available_arms_index)
        r = rewards[action2,t]
        instance2.update_stats(action2, r)
        reward2[t] = np.sum(r)

        action3 = instance3.draw_action(available_arms_index)
        r = rewards[action3,t]
        instance3.update_stats(action3, r)
        reward3[t] = np.sum(r)
        
        
    cumureward = np.cumsum(reward)
    cumureward2 = np.cumsum(reward2)
    cumureward3 = np.cumsum(reward3)

    for t in range(T):
        time_ave_reward[t] = cumureward[t]/(t+1)
        time_ave_reward2[t] = cumureward2[t]/(t+1)
        time_ave_reward3[t] = cumureward3[t]/(t+1)

    l1 = plt.plot(time_ave_reward, label = 'CombTS_Basic (Beta)')
    l2 = plt.plot(time_ave_reward2, label = 'CombTS_Basic (Gaussian)')
    l3 = plt.plot(time_ave_reward3, label = 'CombTS_Single (Gaussian)')

    plt.legend()

# %%