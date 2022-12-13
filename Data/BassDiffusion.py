"""A basic Bass diffusion model class"""

import numpy as np
import matplotlib.pyplot as plt

class BassDiffusion():

    def __init__(self, p, q, m, t_0 = 0.0):
        self.p = p      # rate of innovation
        self.q = q      # rate of imitation
        self.m = m  	# market potential
        self.t_0 = t_0  # starting period of diffusion (defaults to zero)

    def __repr__(self):
        return f"A Bass diffusion model with parameters:\n  p =   {self.p}\n  q =   {self.q}\n  t_0 = {self.t_0}\n  m =   {self.m}"

    # Fraction adopted at beginning of period t
    def F(self, t):
        cur_t = t - self.t_0 # current period, relative to starting period
        if cur_t >= 0.0: # diffusion has started
            return (1.0 - np.exp(-(self.p+self.q)*cur_t))/(1.0 + (self.q / self.p) * np.exp(-(self.p + self.q)*cur_t))           
        else: # diffusion hasn't started yet
            return 0.0


    # Fraction not adopted at beginning of period t
    def S(self, t):
        return 1.0 - self.F(t) # by definition
    
    # Marginal number of sales at time period t (rate of change of A(t))
    # Note: doesn't make too much sense in a discrete setting
    def s(self, t):
        cur_t = t - self.t_0 # current period, relative to starting period
        if cur_t >= 0.0:
            return self.m * ((self.p + self.q)**2 / self.p) * ( np.exp(-(self.p + self.q) * cur_t) / (1.0 + (self.q / self.p) * np.exp(-(self.p + self.q) * cur_t))**2 )
        else:
            return 0.0

    # Amount adopted at beginning of period t
    def A(self, t):
        return self.m * self.F(t)

    # Plot adoption amount over time
    def plot_A(self, t_vec = range(2010, 2070), filename = "", show = True):
        A_vec = [self.A(t_vec[i]) for i in range(len(t_vec))]
        plt.plot(t_vec, A_vec)
        for y in [2020, 2025, 2030, 2040, 2050]:
            plt.plot([y,y], [0,self.m], color='k')
        if filename != "":
            plt.savefig(f"Plots/{filename}")
        if show:
            plt.show()

bassie = BassDiffusion(0.2, 0.2, 100, 2020)

