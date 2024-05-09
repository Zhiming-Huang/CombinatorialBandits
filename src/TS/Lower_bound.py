#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:45:35 2024

@author: Zhiming Huang
"""

import numpy as np
import matplotlib.pyplot as plt

class Lower_bound():
    #implement expTS_Plus algorithm
    def __init__(self, K, u, m = 1, bernoulli = True, gap_deps=True):
        self.gap_deps = gap_deps
        # initialize the action set
       # self.actions = actions
        self.K = K
        # initialize the true mean rewards
        self.u = u
        self.m = m
        #calculate the reward gap between arm 1 and rest of the arms
        self.delta = np.zeros(K)
        for i in range(K):
            self.delta[i] = u[0] - u[i]

        self.bernoulli = bernoulli

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
        
    def lower_bound(self, t):
        if self.gap_deps:
            return self.lower_bound_gap_dep(t)
        else:
            return self.lower_bound_gap_indep(t)

    def lower_bound_gap_indep(self, t):
        #calculate the lower bound based on the formula: \sum_{i=2}^{K} \frac{\log t}{KL(\mu_1, \mu_i)}
        return np.sqrt(self.m * self.K * t * np.log(self.K/self.m))
    
    def lower_bound_gap_dep(self, t):
        #calculate the lower bound based on the formula: \sum_{i=2}^{K} \frac{\log t}{KL(\mu_1, \mu_i)}
        if t < 2:
            return 1
        
        lb = 2
        maxu = max(self.u)
        for i in range(1, self.K):
            if maxu == self.u[i]:
                lb += 0
            else:
                lb += (maxu - self.u[i]) * np.log(t)/self.kl_divergence(maxu, self.u[i])
        return lb


if __name__ == "__main__":
    # module test program
    my_generator = np.random.default_rng()
    T = 1000
    #generate a multivariable gaussian distribution  

    instance = Lower_bound(2, [0.9, 0.8, 0.8, 0.6, 0.5], m = 3)
    
    reward = np.zeros(T)

    for t in range(T):
        reward[t] = instance.lower_bound(t)

    l1 = plt.plot(reward, label = 'Lower_bound')

    plt.legend()

# %%