#!/usr/bin/env python3

import brian2
from brian2 import NeuronGroup, SpikeMonitor, StateMonitor
from brian2 import mV, ms, Gohm, second, ms
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

u_ = np.linspace(u_reset, theta_rh+10*mV, 10000000)

du_dt = (-(u_ - E_L) + delta_T*np.exp((u_ - theta_rh)/delta_T) + R_m*I_rh) / tau_m

plt.figure()
plt.plot(u_/mV, du_dt)
plt.grid()
plt.savefig('phase_diagram.pdf')

def run_sim(initial_state):
    t_sim = .5 * second
    dt = .1 * ms
    brian2.defaultclock.dt = dt

    eqs = '''
        du / dt = (-(u - E_L) + delta_T*exp((u - theta_rh)/delta_T) + R_m * I_rh) / tau_m : volt
    '''

    theta_reset = theta_rh + 10*mV
    neuron = NeuronGroup(1, eqs, threshold='u>theta_reset', reset='u = u_reset', method='euler', refractory=5*ms)
    neuron.u = initial_state

    state_mon = StateMonitor(neuron, 'u', record=0)  # monitor membrane potential of 1st neuron

    spike_mon = SpikeMonitor(neuron)  # record neuron spikes

    brian2.run(t_sim)

    return state_mon.t, state_mon.u[0]


# simulate neuron in different conditions


[t, state1] = run_sim(theta_rh - 1*mV)
[_, state2] = run_sim(theta_rh + 1*mV)
[_, state3] = run_sim(theta_rh + 0.1*mV)


# plot time course of u(t) in different conditions

plt.figure()
plt.subplot(3,1,1)
plt.plot(t/ms, state1 * 1E3)
plt.subplot(3,1,2)
plt.plot(t/ms, state2  * 1E3)
plt.subplot(3,1,3)
plt.plot(t/ms, state3 * 1E3)
plt.savefig('simulations.pdf')

plt.show()  # avoid having multiple plt.show()s in your code
