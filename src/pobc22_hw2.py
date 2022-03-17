#!/usr/bin/env python3

import brian2
from brian2 import NeuronGroup, SpikeMonitor, StateMonitor
from brian2 import mV, ms, Gohm
import matplotlib.pyplot as plt
import numpy as np

# parameters

tau_m = 20 * ms
R_m = 0.02 * Gohm
E_L = -65 * mV
u_reset = -80 * mV
theta_rh = -50 * mV
delta_T = 3 * mV


# calculate rheobase current

I_rh = (theta_rh - E_L - delta_T) / R_m


# plot phase diagram at rheobase
# don't forget to label all axes correctly

u = np.linspace(u_reset, theta_rh+10*mV, 10000000)

du_dt = (-(u - E_L) + delta_T*np.exp((u - theta_rh)/delta_T) + R_m*I_rh) / tau_m

plt.figure()
plt.plot(u, du_dt)
plt.show()

# plt.savefig('phase_diagram.pdf')


# simulate neuron in different conditions

# ...

# neuron = NeuronGroup(..., method='euler')

# brian2.run(...)


# plot time course of u(t) in different conditions

# plt.figure()

...

# plt.savefig('simulations.pdf')

# plt.show()  # avoid having multiple plt.show()s in your code
