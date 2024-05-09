#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:45:35 2024

@author: Zhiming Huang
"""

import numpy as np
import matplotlib.pyplot as plt

class KL_UCB_Plus:
    #implement the KL-UCB algorithm
    def __init__(self, K, T, rnd_generator, bernoulli = True):
        # initialize the action set
       # self.actions = actions
        self.K = K
        self.T = T
        # initialize the probabiltiy distribution
        # self.eta = np.log(K)/t
        self.t = 1
        
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
    
        
    def kl_bern(self, x, y):
        # Kullback-Leibler divergence for Bernoulli distributions. https://en.wikipedia.org/wiki/Bernoulli_distribution#Kullback.E2.80.93Leibler_divergence
        eps = 1e-15
        x = min(max(x, eps), 1 - eps)
        y = min(max(y, eps), 1 - eps)
        return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))    

 
    def kl_Gauss(self, x, y, sig2x=0.25, sig2y=None):
        eps = 1e-15
        # Kullback-Leibler divergence for Gaussian distributions. https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Examples
        if sig2y is None or - eps < (sig2y - sig2x) < eps:
            return (x - y) ** 2 / (2. * sig2x)
        else:
            return (x - y) ** 2 / (2. * sig2y) + 0.5 * ((sig2x/sig2y)**2 - 1 - np.log(sig2x/sig2y))

    def kl_divergence(self, x, y):
        if self.bernoulli:
            return self.kl_bern(x, y)
        else:
            return self.kl_Gauss(x, y)

    
    def draw_action(self):
        # if there is still an arm not played, play it
        if self.t <= self.K:
            return self.t - 1
        #draw a sample from the kL laplace distribution
        samples = np.zeros(self.K)
        max_iterations = 100
        for i in range(self.K):
            n = self.alpha[i]+self.beta[i]
            #p = self.rnd_generator.uniform(0,1)
            u = self.alpha[i]/n
            samples[i] = u
            m = u
            d = max(0, np.log(self.T/(self.K*n)*(max(0, np.log(self.T/(self.K*n))**2) + 1)))
            _count_iteration = 0
            precision=1e-6
            upper = 1 - 1e-15
            while _count_iteration < max_iterations and upper - samples[i] > precision:
                _count_iteration += 1
                m = (samples[i] + upper) / 2.
                if self.kl_divergence(upper, m) > d/n:
                    upper = m
                else:
                    samples[i] = m

        return np.argmax(samples)
        
    def update_stats(self, action, r):
        # update talpha and beta
        self.alpha[action] += r
        self.beta[action] += 1 - r
        self.t += 1

# #implement a class KL_UCB which is a subclass of KL_UCB_Plus
class KL_UCB(KL_UCB_Plus):
    def __init__(self, K, T, rnd_generator, bernoulli = True):
        super().__init__(K, T, rnd_generator, bernoulli)
    
    def draw_action(self):
        # if there is still an arm not played, play it
        if self.t <= self.K:
            return self.t - 1
        #draw a sample from the kL laplace distribution
        samples = np.zeros(self.K)

        max_iterations = 100
        for i in range(self.K):
            n = self.alpha[i]+self.beta[i]
            p = self.rnd_generator.uniform(0,1)
            u = self.alpha[i]/n
            samples[i] = u
            m = u
            upper = 1
            d = np.log(t) + 3*np.log(np.log(t))
            _count_iteration = 0
            precision=1e-6
            while _count_iteration < max_iterations and u - samples[i] > precision:
                _count_iteration += 1
                m = (samples[i] + upper) / 2.
                if self.kl_divergence(u, m) > d/n:
                    upper = m
                else:
                    samples[i] = m

        return np.argmax(samples)


if __name__ == "__main__":
    # module test program
    my_generator = np.random.default_rng()

    T = 1000
    instance = KL_UCB(2, T, my_generator)
    
    #generate a multivariable gaussian distribution  
    lowarm = my_generator.uniform(0.0,0.4,[1, T])
    higharm = my_generator.uniform(0.8,1.0,[1, T])

    #generate bernoulli random variables based on lowarm and higharm
    lowarm = my_generator.binomial(1,lowarm)
    higharm = my_generator.binomial(1,higharm)

    rewards = np.vstack([higharm, lowarm])
    
    reward = np.zeros(T)
    time_ave_reward = np.zeros(T)

    for t in range(T):
        action = instance.draw_action()
        r = rewards[action,t]
        instance.update_stats(action, r)
        reward[t] = r
        
        
    cumureward = np.cumsum(reward)
    opt = max(np.sum(rewards,axis = 1)/T)
    for t in range(T):
        time_ave_reward[t] = cumureward[t]/(t+1)

    l1 = plt.plot(time_ave_reward, label = 'KL_UCB')

    plt.legend()

# %%