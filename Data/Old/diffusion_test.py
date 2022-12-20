"""A basic Bass diffusion model to test and plot some values"""

import numpy as np
import matplotlib.pyplot as plt

class BassDiffusion():

    def __init__(self, p, q, m, t_0 = 0.0):
        self.p = p      # rate of innovation
        self.q = q      # rate of imitation
        self.m = m  	# market potential
        self.t_0 = t_0  # starting period of diffusion (defaults to zero)

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

    def plot_A(self, t_vec = range(2010, 2070), filename = ""):
        A_vec = [self.A(t_vec[i]) for i in range(len(t_vec))]
        plt.plot(t_vec, A_vec)
        for y in [2022, 2026, 2030, 2040, 2050]:
            plt.plot([y,y], [0,m], color='k')
        if filename != "":
            plt.savefig(f"Plots/{filename}")
        plt.show()

    

# Initialize parameters (base on wikipedia values)
p = 0.03    # rate of innovation
q = 0.38    # rate of imitation
m = 100.0   # market potential (in our case study this should be as a percentage of the total vehicles of that mode)
t_0 = 2030  # period in which diffusion starts

# create model
bass = BassDiffusion(p, q, m, t_0)

# create a vector
t_vec = range(2010, 2070)
A_vec = [bass.A(t_vec[i]) for i in range(len(t_vec))]

# plot diffusion process
plt.plot(t_vec, A_vec)

# add 90% line
p_90_vec = [0.9 * bass.m] * len(t_vec)
plt.plot(t_vec, p_90_vec)

# show plot
plt.show()

#####

# test multiple q values
p = 0.02
t_0 = 2025
m = 10
for q in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    bass = BassDiffusion(p, q, m, t_0)
    A_vec = [bass.A(t_vec[i]) for i in range(len(t_vec))]
    plt.plot(t_vec, A_vec)

for y in [2020, 2025, 2030, 2040, 2050]:
    plt.plot([y,y], [0,m], color='k')
#plt.plot(t_vec, p_90_vec)
plt.show()


