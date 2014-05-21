"""
Test of the EIF_cond_alpha_isfa_ista model

Andrew Davison, UNIC, CNRS
December 2007

"""

from pyNN.utility import get_simulator, normalized_filename

sim, options = get_simulator(("--plot-figure",
                              "Plot the simulation results to a file."))

sim.setup(timestep=0.01, min_delay=0.1, max_delay=4.0)

cell_type = sim.EIF_cond_alpha_isfa_ista(i_offset=1.0, tau_refrac=2.0, v_spike=-40)
ifcell = sim.create(cell_type)
print ifcell[0].get_parameters()

filename = normalized_filename("Results", "EIF_cond_alpha_isfa_ista", "pkl",
                               options.simulator)
sim.record('v', ifcell, filename, annotations={'script_name': __file__})
sim.run(200.0)

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    data = ifcell.get_data().segments[0]
    vm = data.filter(name="v")[0]
    Figure(
        Panel(vm, ylabel="Membrane potential (mV)", xlabel="Time (ms)",
              xticks=True),
        title=__file__,
    ).save(options.plot_figure)

sim.end()
