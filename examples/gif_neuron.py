"""
todo: write docstring



"""

import matplotlib
matplotlib.use('Agg')
from numpy import arange
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
        'i_offset':     0.0,  # Offset current in nA
        'v_t_star':   -55.0,  # Threshold baseline in mV.
        'lambda0':      1.0,  # Firing intensity at threshold in Hz.
        'tau_eta1':     1.0,  # }
        'tau_eta2':    10.0,  # } Time constants for spike-triggered current in ms.
        'tau_eta3':   100.0,  # }
        'tau_gamma1':   1.0,  # }
        'tau_gamma2':  10.0,  # } Time constants for spike-frequency adaptation in ms.
        'tau_gamma3': 100.0,  # }
        # the following parameters have different values for each neuron
        'delta_v':  [1e-6, 1e-6, 0.5, 0.5],  # Threshold sharpness in mV.
        'a_eta1':   [0.1, 0.0, 0.0, 0.0],  # }
        'a_eta2':   [0.1, 0.0, 0.0, 0.0],  # } Post-spike increments for spike-triggered current in nA
        'a_eta3':   [0.1, 0.0, 0.0, 0.0],  # }
        'a_gamma1': [0.0, 5.0, 0.0, 0.0],  # }
        'a_gamma2': [0.0, 5.0, 0.0, 0.0],  # } Post-spike increments for spike-frequency adaptation in mV
        'a_gamma3': [0.0, 5.0, 0.0, 0.0],  # }
    },
    'stimulus': {
        'start': 20.0,
        'stop': t_stop - 20.0,
        'amplitude': 0.6
    }
}

neurons = sim.Population(4, sim.GIF_cond_exp(**parameters['neurons']))

electrode = sim.DCSource(**parameters['stimulus'])
electrode.inject_into(neurons)

neurons.record(['v'])  #, 'E_sfa'])
#neurons.initialize(v=-70.0)


# === Run the simulation =====================================================

sim.run(t_stop)

#import nest
#print(nest.GetStatus(neurons.all_cells.tolist()))

# === Save the results, optionally plot a figure =============================

filename = normalized_filename("Results", "gif_neuron", "pkl",
                               options.simulator, sim.num_processes())
#filename = "Results/gif_neuron.pkl"
neurons.write_data(filename, annotations={'script_name': __file__})

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    figure_filename = filename.replace("pkl", "png")
    data = neurons.get_data().segments[0]
    v = data.filter(name="v")[0]
    #u = data.filter(name="u")[0]
    Figure(
        Panel(v, ylabel="Membrane potential (mV)", xticks=True,
              xlabel="Time (ms)", yticks=True),
        #Panel(u, ylabel="u variable (units?)"),
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename)
    print(figure_filename)


# === Clean up and quit ========================================================

sim.end()
