import mlx.core as mx
import matplotlib.pyplot as plt
import numpy as np

class CombTS_Basic:
    #implement Combinatorial Thompson sampling for Beta distribution or Normal distribution
    def __init__(self, m, K, bernoulli = True, beta = 0.01):
        # initialize the action set
       # self.actions = actions
        self.m = m  
        self.K = K
        # initialize the probabiltiy distribution
        # self.eta = np.log(K)/t
        self.t = 1
        
        #initialize the parameters for beta priors
        self.alpha = mx.zeros(K)
        self.beta = mx.zeros(K)

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
                    estimate_mean[k] = self.rnd_generator.normal(0, np.sqrt(self.m*np.log(self.t))/np.sqrt(self.alpha[k]+self.beta[k]+1))
                else:
                    estimate_mean[k] = self.rnd_generator.normal(self.alpha[k]/(self.alpha[k]+self.beta[k]), np.sqrt(0.01*self.m*np.log(self.t)/(self.alpha[k]+self.beta[k]+1)))
        
        ind = np.argpartition(estimate_mean[available_arms], -self.m)[-self.m:]
        return available_arms[ind]
            

    def update_stats(self, action, r):
        # update talpha and beta
        self.alpha[action] += r
        self.beta[action] += 1 - r
        self.t += 1
