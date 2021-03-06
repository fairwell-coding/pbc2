#!/usr/bin/env python3

import brian2
from brian2 import NeuronGroup, SpikeMonitor, StateMonitor, namp
from brian2 import mV, ms, Gohm, second, ms
import matplotlib.pyplot as plt
import numpy as np

# parameters
tau_m = 20 * ms  # membrane time constant
R_m = 0.02 * Gohm  # membrane resistance
E_L = -65 * mV  # reversal potential of leak current
u_reset = -80 * mV  # membrane reset potential
theta_rh = -50 * mV  # threshold at rheobase
delta_T = 3 * mV  # steepness of action potential
sigma_u = 1 * mV  # standard deviation for added noise

# define rheobase current
I_rh = (theta_rh - E_L - delta_T) / R_m  # analytically calculated rheobase current
rheobase_delta = 0.04 * namp
I_rh += rheobase_delta  # add a very small additional positive current to lift the minimum above 0 to avoid the saddle point


def __task_2a():
    """ task 2a: Plot phase diagram for analytically calculated rheobase current
    """

    u_ = np.linspace(u_reset, theta_rh + 10 * mV, 10000000)
    du_dt = (-(u_ - E_L) + delta_T * np.exp((u_ - theta_rh) / delta_T) + R_m * I_rh) / tau_m

    fig, ax = plt.subplots()
    plt.plot(u_ / mV, du_dt, label="Analytically calculated rheobase current")
    plt.grid()
    plt.axhline(color="darkred", linestyle="--", label="Minimum threshold required for rheobase current")
    ax.set_title('task 2a: Phase diagram')
    ax.legend(loc='upper left')
    ax.set_xlabel('u(t) [in mV]')
    ax.set_ylabel('du / dt')

    plt.savefig('phase_diagram.pdf')


def run_sim(initial_state, eqs):
    t_sim = .5 * second
    dt = .1 * ms
    brian2.defaultclock.dt = dt

    theta_reset = theta_rh + 10*mV
    neuron = NeuronGroup(1, eqs, threshold='u>theta_reset', reset='u = u_reset', method='euler', refractory=5*ms)
    neuron.u = initial_state

    state_mon = StateMonitor(neuron, 'u', record=0)  # monitor membrane potential of 1st neuron
    spike_mon = SpikeMonitor(neuron)  # record neuron spikes

    brian2.run(t_sim)

    return state_mon.t/ms, state_mon.u[0]


def __task_2b():
    """ task 2b: Calculate and plot membrane potentials for different initial membrane potentials.
    """

    eqs = '''
        du / dt = (-(u - E_L) + delta_T * exp((u - theta_rh)/delta_T) + R_m * I_rh) / tau_m : volt
    '''

    # Calculate membrane potentials for different starting conditions
    [time, state1] = run_sim(theta_rh - 1 * mV, eqs)
    [_, state2] = run_sim(theta_rh + 1 * mV, eqs)
    [_, state3] = run_sim(theta_rh + 0.1 * mV, eqs)

    # plots
    fig, ax = plt.subplots(3)
    fig.suptitle('task 2b: Comparison of starting potentials', fontsize=16)

    ax[0].plot(time, state1 * 1E3, color="forestgreen", label="(i) u_0 = theta_rh - 1mV")
    ax[0].legend(loc='lower right')
    ax[0].set_xlabel('t / ms')
    ax[0].set_ylabel('u(t) / mV')

    ax[1].plot(time, state2 * 1E3, color="forestgreen", label="(ii) u_0 = theta_rh + 1mV")
    ax[1].legend(loc='lower right')
    ax[1].set_xlabel('t / ms')
    ax[1].set_ylabel('u(t) / mV')
    ax[1].set_yticks([-80, -60, -40])

    ax[2].plot(time, state3 * 1E3, color="forestgreen", label="(iii) u_0 = theta_rh + 0.1mV")
    ax[2].legend(loc='lower right')
    ax[2].set_xlabel('t / ms')
    ax[2].set_ylabel('u(t) / mV')

    plt.subplots_adjust(top=0.8, hspace=0.7)
    plt.savefig('simulations.pdf')


def __task_2c():
    """ task 2c: Observe model behaviour by adding Gaussian noise. Note that each run results in a different result/plot due to its stochastic behaviour.
    """

    eqs = '''
        du / dt = (-(u - E_L) + delta_T * exp((u - theta_rh)/delta_T) + R_m * I_rh) / tau_m + sigma_u * sqrt (2 / tau_m ) * xi : volt
    '''

    # Calculate membrane potentials for different starting conditions with added noise
    [time, state1] = run_sim(theta_rh - 1 * mV, eqs)
    [_, state2] = run_sim(theta_rh + 1 * mV, eqs)
    [_, state3] = run_sim(theta_rh + 0.1 * mV, eqs)

    # plots
    fig, ax = plt.subplots(3)
    fig.suptitle('task 2c: Comparison of starting potentials with noise', fontsize=16)

    ax[0].plot(time, state1 * 1E3, color="forestgreen", label="(i) u_0 = theta_rh - 1mV")
    ax[0].legend(loc='lower right')
    ax[0].set_xlabel('t / ms')
    ax[0].set_ylabel('u(t) / mV')

    ax[1].plot(time, state2 * 1E3, color="forestgreen", label="(ii) u_0 = theta_rh + 1mV")
    ax[1].legend(loc='lower right')
    ax[1].set_xlabel('t / ms')
    ax[1].set_ylabel('u(t) / mV')
    ax[1].set_yticks([-80, -60, -40])

    ax[2].plot(time, state3 * 1E3, color="forestgreen", label="(iii) u_0 = theta_rh + 0.1mV")
    ax[2].legend(loc='lower right')
    ax[2].set_xlabel('t / ms')
    ax[2].set_ylabel('u(t) / mV')


def __task_2c_3():
    """ Use same noise input as in task 2c but with external current I(t) = I_rh = 0 and initial membrane potential u_0 = E_L.
    """

    eqs = '''
        du / dt = (-(u - E_L) + delta_T * exp((u - theta_rh)/delta_T)) / tau_m + sigma_u * sqrt (2 / tau_m ) * xi : volt
    '''

    # Calculate membrane potential with I_rh = 0 and u_0 = E_L
    [time, state] = run_sim(E_L, eqs)

    # plot
    fig, ax = plt.subplots()
    fig.suptitle('task 2c - part III', fontsize=16)

    ax.plot(time, state * 1E3, color="forestgreen", label="I(t) = 0 and u_0 = E_L")
    ax.legend(loc='lower right')
    ax.set_xlabel('t / ms')
    ax.set_ylabel('u(t) / mV')


__task_2a()
__task_2b()
__task_2c()
__task_2c_3()

plt.show()
