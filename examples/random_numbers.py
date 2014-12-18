"""
An example to illustrate random number handling in PyNN, in particular
the difference between "native" and Python random number generators.
"""

import numpy
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility import get_simulator

sim, options = get_simulator(
                    ("--plot-figure", "plot the simulation results to a file", {"action": "store_true"}),
                    ("--debug", "print debugging information"))

sim.setup()

python_rng = NumpyRNG(seed=98497627)
native_rng = sim.NativeRNG(seed=87354762)

cell_type = sim.IF_cond_exp(tau_m=RandomDistribution('normal', (15.0, 2.0), rng=python_rng))  # not possible with NEST to use NativeRNG here
v_init = RandomDistribution('uniform',
                            (cell_type.default_parameters['v_rest'], cell_type.default_parameters['v_thresh']),
                            rng=python_rng)  # not possible with NEST to use NativeRNG here

p1 = sim.Population(10, sim.SpikeSourcePoisson(rate=100.0))  # in the current version, can't specify the RNG - it is always native
p2 = sim.Population(10, cell_type, initial_values={'v': v_init})

p1.record("spikes")
p2.record("spikes")
p2.sample(3, rng=python_rng).record("v")  # can't use native RNG here

connector_native = sim.FixedProbabilityConnector(p_connect=0.7, rng=native_rng)
connector_python = sim.FixedProbabilityConnector(p_connect=0.7, rng=python_rng)

synapse_type_native = sim.StaticSynapse(weight=RandomDistribution('gamma', k=2.0, theta=0.5, rng=native_rng),
                                        delay=0.5)
synapse_type_python = sim.StaticSynapse(weight=RandomDistribution('gamma', k=2.0, theta=0.5, rng=python_rng),
                                        delay=0.5)

projection_native = sim.Projection(p1, p2, connector_native, synapse_type_native)
projection_python = sim.Projection(p1, p2, connector_python, synapse_type_python)

weights_python = projection_python.get("weight", format="array")
weights_native = projection_native.get("weight", format="array")
print(weights_python)
print(weights_native)

sim.run(100.0)

sim.end()

if options.plot_figure:
    from pyNN.utility import normalized_filename
    from pyNN.utility.plotting import Figure, Panel
    filename = normalized_filename("Results", "random_numbers", "png", options.simulator)
    weights_python[numpy.isnan(weights_python)] = 0
    weights_native[numpy.isnan(weights_native)] = 0
    Figure(
        Panel(weights_python, cmap='gray_r', xlabel="Python RNG"),
        Panel(weights_native, cmap='gray_r', xlabel="Native RNG"),
    ).save(filename)
    print(filename)