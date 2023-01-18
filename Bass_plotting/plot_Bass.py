"""Making an example plot of the Bass diffusion process"""

import matplotlib.pyplot as plt
import numpy as np
from Data.BassDiffusion import BassDiffusion 


p_base = 0.015
q_base = 0.15
start_year = 2025

bassie_base = BassDiffusion(p_base, q_base, 100, t_0=start_year)
bassie_slow = BassDiffusion(0.5*p_base, 0.5*q_base, 100, t_0=start_year)
bassie_fast = BassDiffusion(1.5*p_base, 1.5*q_base, 100, t_0=start_year)

A_base_2030 = bassie_base.A(2030)

t_vec = range(2020, 2050)

A_base = np.zeros(len(t_vec))
A_slow = np.zeros(len(t_vec))
A_fast = np.zeros(len(t_vec))

for i in range(len(t_vec)):
    A_base[i] = bassie_base.A(t_vec[i])

    if t_vec[i] <= 2030:
        A_slow[i] = A_base[i]
        A_fast[i] = A_base[i]
    else:
        A_slow[i] = bassie_slow.A_from_starting_point(t_vec[i], A_base_2030, 2030)
        A_fast[i] = bassie_fast.A_from_starting_point(t_vec[i], A_base_2030, 2030)

plt.plot(t_vec, A_base, label = "base", zorder = 3)
plt.plot(t_vec, A_slow, label = "slow", zorder = 2)
plt.plot(t_vec, A_fast, label = "fast", zorder = 1)

plt.ylabel("Adoption level")

plt.legend()

plt.savefig("Plots/Bass/Bass_example.png")

plt.show()



