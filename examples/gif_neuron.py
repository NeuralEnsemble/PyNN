"""
Demonstration of the Generalized Integrate-and-Fire model described by Pozzorini et al. (2015)

We simulate four neurons with different parameters:
    1. spike-triggered current, fixed threshold, deterministic spiking
    2. no spike-triggered current, dynamic threshold, deterministic spiking
    3 & 4. no spike-triggered current, fixed threshold, stochastic spiking

Since neurons 1 and 2 have deterministic spiking, they should produce the same spike times with
different simulators. Neurons 3 and 4, being stochastic, should spike at different times.

Reference:

Pozzorini, Christian, Skander Mensi, Olivier Hagens, Richard Naud, Christof Koch, and Wulfram Gerstner (2015)
"Automated High-Throughput Characterization of Single Neurons by Means of Simplified Spiking Models."
PLOS Comput Biol 11 (6): e1004275. doi:10.1371/journal.pcbi.1004275.

"""

import matplotlib
matplotlib.use('Agg')
from pyNN.utility import get_simulator, init_logging, normalized_filename


# === Configure the simulator ================================================

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

sim.setup(timestep=0.01, min_delay=1.0)


# === Build and instrument the network =======================================

t_stop = 300.0

parameters = {
    'neurons': {
        'v_rest':     -65.0,  # Resting membrane potential in mV.
        'cm':           1.0,  # Capacity of the membrane in nF
        'tau_m':       20.0,  # Membrane time constant in ms.
        'tau_refrac':   4.0,  # Duration of refractory period in ms.
        'tau_syn_E':    5.0,  # Decay time of the excitatory synaptic conductance in ms.
        'tau_syn_I':    5.0,  # Decay time of the inhibitory synaptic conductance in ms.
        'e_rev_E':      0.0,  # Reversal potential for excitatory input in mV
        'e_rev_I':    -70.0,  # Reversal potential for inhibitory input in mV
        'v_reset':    -65.0,  # Reset potential after a spike in mV.
        'i_offset':     [0.0, 0.0, 0.0, 0.0],  # Offset current in nA
        'v_t_star':   -55.0,  # Threshold baseline in mV.
        'lambda0':      1.0,  # Firing intensity at threshold in Hz.
        'tau_eta':   (1.0, 10.0, 100.0),  # Time constants for spike-triggered current in ms.
        'tau_gamma': (1.0, 10.0, 100.0),  # Time constants for spike-frequency adaptation in ms.
        # the following parameters have different values for each neuron
        'delta_v':   [1e-6, 1e-6, 0.5, 0.5],  # Threshold sharpness in mV.
        'a_eta':     [(0.1, 0.1, 0.1),  # Post-spike increments for spike-triggered current in nA
                      (0.0, 0.0, 0.0),
                      (0.0, 0.0, 0.0),
                      (0.0, 0.0, 0.0)],
        'a_gamma':   [(0.0, 0.0, 0.0),  # Post-spike increments for spike-frequency adaptation in mV
                      (5.0, 5.0, 5.0),
                      (0.0, 0.0, 0.0),
                      (0.0, 0.0, 0.0)],
    },
    'stimulus': {
        'start': 20.0,
        'stop': t_stop - 20.0,
        'amplitude': 0.6
    }
}

neurons = sim.Population(4, sim.GIF_cond_exp(**parameters['neurons']),
                         initial_values={'v': -65.0, 'v_t': -55.0})

print("i_offset = ", neurons.get('i_offset'))
print("v_t_star = ", neurons.get('v_t_star'))
print("delta_v = ", neurons.get('delta_v'))
print("tau_eta = ", neurons.get('tau_eta'))
print("a_gamma = ", neurons.get('a_gamma'))

electrode = sim.DCSource(**parameters['stimulus'])
electrode.inject_into(neurons)

neurons.record(['v', 'i_eta', 'v_t'])


# === Run the simulation =====================================================

sim.run(t_stop)

# === Save the results, optionally plot a figure =============================

filename = normalized_filename("Results", "gif_neuron", "pkl",
                               options.simulator, sim.num_processes())
neurons.write_data(filename, annotations={'script_name': __file__})

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    figure_filename = filename.replace("pkl", "png")
    data = neurons.get_data().segments[0]
    v = data.filter(name="v")[0]
    v_t = data.filter(name="v_t")[0]
    i_eta = data.filter(name="i_eta")[0]
    Figure(
        Panel(v, ylabel="Membrane potential (mV)",
              yticks=True, ylim=[-66, -52]),
        Panel(v_t, ylabel="Threshold (mV)",
              yticks=True),
        Panel(i_eta, ylabel="i_eta (nA)", xticks=True,
              xlabel="Time (ms)", yticks=True),
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename)
    print(figure_filename)


# === Clean up and quit ========================================================

sim.end()
