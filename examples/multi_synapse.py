"""
A demonstration of the use of the PointNeuron model, which allows the composition
of any neuron model with an unlimited number of different synapse models
(although not all combinations will be available on all backend simulators).

Usage: multi_synapse.py [-h] [--plot-figure] [--debug DEBUG] simulator

positional arguments:
  simulator      neuron, nest, brian or another backend simulator

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  Plot the simulation results to a file.
  --debug DEBUG  Print debugging information

"""


from pyNN.parameters import Sequence
from pyNN.utility import get_simulator, init_logging, normalized_filename


# === Configure the simulator ================================================

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

sim.setup(timestep=0.1, min_delay=1.0)


# === Build and instrument the network =======================================

celltype = sim.PointNeuron(
                sim.AdExp(tau_m=10.0, v_rest=-60.0),
                AMPA=sim.AlphaPSR(tau_syn=1.0, e_syn=0.0),
                NMDA=sim.AlphaPSR(tau_syn=20.0, e_syn=0.0),
                GABAA=sim.AlphaPSR(tau_syn=1.5, e_syn=-70.0),
                GABAB=sim.AlphaPSR(tau_syn=15.0, e_syn=-90.0))

neurons = sim.Population(1, celltype, initial_values={'v': -60.0})

neurons.record(['v', 'AMPA_gsyn', 'NMDA_gsyn', 'GABAA_gsyn', 'GABAB_gsyn'])  #, 'AMPA.gsyn', 'NMDA.gsyn', 'GABAA.gsyn', 'GABAB.gsyn'])

print("tau_m = ", neurons.get("tau_m"))
print("GABAA.e_syn = ", neurons.get("GABAA.e_syn"))

inputs = sim.Population(4,
                        sim.SpikeSourceArray(spike_times=[
                            Sequence([30.0]),
                            Sequence([60.0]),
                            Sequence([90.0]),
                            Sequence([120.0])])
                        )

connections = {
    "AMPA": sim.Projection(inputs[0:1], neurons, sim.OneToOneConnector(),
                           synapse_type=sim.StaticSynapse(weight=0.01, delay=1.5),
                           receptor_type="AMPA", label="AMPA"),
    "GABAA": sim.Projection(inputs[1:2], neurons, sim.OneToOneConnector(),
                            synapse_type=sim.StaticSynapse(weight=0.1, delay=1.5),
                            receptor_type="GABAA", label="GABAA"),
    "NMDA": sim.Projection(inputs[2:3], neurons, sim.OneToOneConnector(),
                           synapse_type=sim.StaticSynapse(weight=0.005, delay=1.5),
                           receptor_type="NMDA", label="NMDA"),
}
# === Run the simulation =====================================================

sim.run(200.0)


# === Save the results, optionally plot a figure =============================

#filename = normalized_filename("Results", "multi_synapse", "pkl", options.simulator)
filename = "Results/multi_synapse_{}.pkl".format(options.simulator)
data = neurons.get_data().segments[0]

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    figure_filename = filename.replace("pkl", "png")
    Figure(
        Panel(data.filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              xticks=True, xlabel="Time (ms)",
              yticks=True), #ylim=(-66, -48)),
        title="Neuron with multiple synapse time constants",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename)
    print(figure_filename)

    figure_filename_cond_ampa = filename.replace(".pkl", "_cond_ampa.png")
    Figure(
         Panel(data.filter(name='AMPA_gsyn')[0],
               xticks=True, xlabel="Time (ms)",
               ylabel="AMPA Conductance (uS)",
               yticks=True),
        title="Neuron with multiple synapse time constants",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename_cond_ampa)
    print(figure_filename_cond_ampa)

    figure_filename_cond_nmda = filename.replace(".pkl", "_cond_nmda.png")
    Figure(
         Panel(data.filter(name='NMDA_gsyn')[0],
               xticks=True, xlabel="Time (ms)",
               ylabel="NMDA Conductance (uS)",
               yticks=True),
        title="Neuron with multiple synapse time constants",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename_cond_nmda)
    print(figure_filename_cond_nmda)

    figure_filename_cond_gabaa = filename.replace(".pkl", "_cond_gabaa.png")
    Figure(
         Panel(data.filter(name='GABAA_gsyn')[0],
               xticks=True, xlabel="Time (ms)",
               ylabel="GABAA Conductance (uS)",
               yticks=True),
        title="Neuron with multiple synapse time constants",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename_cond_gabaa)
    print(figure_filename_cond_gabaa)

    figure_filename_cond_gabab = filename.replace(".pkl", "_cond_gabab.png")
    Figure(
         Panel(data.filter(name='GABAB_gsyn')[0],
               xticks=True, xlabel="Time (ms)",
               ylabel="GABAB Conductance (uS)",
               yticks=True),
        title="Neuron with multiple synapse time constants",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename_cond_gabab)
    print(figure_filename_cond_gabab)


# === Clean up and quit ========================================================

sim.end()
