# encoding: utf-8
"""
Compare the response of IF and HH neurones when they recieve defferent type of current injections

There are four "standard" current sources in PyNN which are shown here:

    - DCSource
    - ACSource
    - StepCurrentSource
    - NoisyCurrentSource

Any other current waveforms can be implemented using StepCurrentSource.


Usage: IF_and_HH_responses_to_injection.py [-h] [--plot-figure]  simulator

positional arguments:
  simulator      neuron, nest, brian or another backend simulator

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  plot the simulation results to a file
  --debug DEBUG  print debugging information

"""


# === Configure the simulator ================================================
from pyNN.utility import get_simulator, normalized_filename
sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file",
                              {"action": "store_true"}))

sim.setup()

# === Create four cells of HH anf IF and inject the 4 types of currents =====================

current_sources = [sim.DCSource(amplitude=1, start=20.0, stop=100.0),
                   sim.StepCurrentSource(times=[20.0, 50.0, 100.0],
                                         amplitudes=[0.4, 0.6, -0.2]),
                   sim.ACSource(start=20.0, stop=100.0, amplitude=0.4,
                                offset=0.1, frequency=10.0, phase=180.0),
                   sim.NoisyCurrentSource(mean=0.5, stdev=0.2, start=20.0,
                                          stop=100.0, dt=1.0)]

hhcell1 = sim.Population(1, sim.HH_cond_exp())
hhcell2 = sim.Population(1, sim.HH_cond_exp())
hhcell3 = sim.Population(1, sim.HH_cond_exp())
hhcell4 = sim.Population(1, sim.HH_cond_exp())

ifcell1 = sim.Population(1, sim.IF_cond_exp())
ifcell2 = sim.Population(1, sim.IF_cond_exp())
ifcell3 = sim.Population(1, sim.IF_cond_exp())
ifcell4 = sim.Population(1, sim.IF_cond_exp())


for hhcell, ifcell, current_source in zip([hhcell1, hhcell2, hhcell3, hhcell4], [ifcell1, ifcell2, ifcell3, ifcell4], current_sources):
    hhcell[0].inject(current_source)
    ifcell[0].inject(current_source)


#prepare the recording
filename = normalized_filename("Results", "current_injection", "pkl", options.simulator)
sim.record('v', ifcell1, filename, annotations={'script_name': __file__})
sim.record('v', ifcell2, filename, annotations={'script_name': __file__})
sim.record('v', ifcell3, filename, annotations={'script_name': __file__})
sim.record('v', ifcell4, filename, annotations={'script_name': __file__})

sim.record('v', hhcell1, filename, annotations={'script_name': __file__})
sim.record('v', hhcell2, filename, annotations={'script_name': __file__})
sim.record('v', hhcell3, filename, annotations={'script_name': __file__})
sim.record('v', hhcell4, filename, annotations={'script_name': __file__})


# === Run the simulation =====================================================
sim.run(300.0)

# === Save the results, optionally plot a figure =============================

vmi1 = ifcell1.get_data().segments[0].filter(name="v")[0]
vmi2 = ifcell2.get_data().segments[0].filter(name="v")[0]
vmi3 = ifcell3.get_data().segments[0].filter(name="v")[0]
vmi4 = ifcell4.get_data().segments[0].filter(name="v")[0]

vmh1 = hhcell1.get_data().segments[0].filter(name="v")[0]
vmh2 = hhcell2.get_data().segments[0].filter(name="v")[0]
vmh3 = hhcell3.get_data().segments[0].filter(name="v")[0]
vmh4 = hhcell4.get_data().segments[0].filter(name="v")[0]

sim.end()

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    from quantities import mV
    figure_filename = filename.replace("pkl", "png")
    Figure(
        Panel( vmi1, vmh1, y_offset=-10 * mV, xticks=True, yticks=True,
            xlabel="Time (ms)", ylabel="Membrane potential (mV)",
            ylim=(-96, -20)),
        Panel( vmi2, vmh2, y_offset=-10 * mV, xticks=True, yticks=True,
            xlabel="Time (ms)", ylabel="Membrane potential (mV)",
            ylim=(-96, -20)),
        Panel( vmi3, vmh3, y_offset=-10 * mV, xticks=True, yticks=True,
            xlabel="Time (ms)", ylabel="Membrane potential (mV)",
            ylim=(-96, -20)),
        Panel( vmi4, vmh4, y_offset=-10 * mV, xticks=True, yticks=True,
            xlabel="Time (ms)", ylabel="Membrane potential (mV)",
            ylim=(-96, -20)),
            title="Different types of current injection in IF(blue) and HH(green) neurons ",
            annotations="Simulated with %s" % options.simulator.upper()
        ).save(figure_filename)









