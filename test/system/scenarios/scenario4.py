

import logging
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.space import Space, Grid3D, RandomStructure, Cuboid
from pyNN.utility import init_logging
from .registry import register


logger = logging.getLogger("TEST")


@register(exclude=["nemo"])
def scenario4(sim):
    """
    Network with spatial structure
    """
    init_logging(logfile=None, debug=True)
    sim.setup()
    rng = NumpyRNG(seed=76454, parallel_safe=False)

    input_layout = RandomStructure(boundary=Cuboid(width=500.0, height=500.0, depth=100.0),
                                   origin=(0, 0, 0), rng=rng)
    inputs = sim.Population(100, sim.SpikeSourcePoisson(rate=RandomDistribution('uniform', low=3.0, high=7.0, rng=rng)),
                            structure=input_layout, label="inputs")
    output_layout = Grid3D(aspect_ratioXY=1.0, aspect_ratioXZ=5.0, dx=10.0, dy=10.0, dz=10.0,
                           x0=0.0, y0=0.0, z0=200.0)
    outputs = sim.Population(200, sim.EIF_cond_exp_isfa_ista(),
                             initial_values={'v': RandomDistribution('normal', mu=-65.0, sigma=5.0, rng=rng),
                                             'w': RandomDistribution('normal', mu=0.0, sigma=1.0, rng=rng)},
                             structure=output_layout,  # 10x10x2 grid
                             label="outputs")
    logger.debug("Output population positions:\n %s", outputs.positions)
    DDPC = sim.DistanceDependentProbabilityConnector
    input_connectivity = DDPC("0.5*exp(-d/100.0)", rng=rng)
    recurrent_connectivity = DDPC("sin(pi*d/250.0)**2", rng=rng)
    depressing = sim.TsodyksMarkramSynapse(weight=RandomDistribution('normal', mu=0.1, sigma=0.02, rng=rng),
                                           delay="0.5 + d/100.0",
                                           U=0.5, tau_rec=800.0, tau_facil=0.0)
    facilitating = sim.TsodyksMarkramSynapse(weight=0.05,
                                             delay="0.2 + d/100.0",
                                             U=0.04, tau_rec=100.0,
                                             tau_facil=1000.0)
    input_connections = sim.Projection(inputs, outputs, input_connectivity,
                                       receptor_type='excitatory',
                                       synapse_type=depressing,
                                       space=Space(axes='xy'),
                                       label="input connections")
    recurrent_connections = sim.Projection(outputs, outputs, recurrent_connectivity,
                                           receptor_type='inhibitory',
                                           synapse_type=facilitating,
                                           space=Space(periodic_boundaries=((-100.0, 100.0), (-100.0, 100.0), None)),  # should add "calculate_boundaries" method to Structure classes
                                           label="recurrent connections")
    outputs.record('spikes')
    outputs.sample(10, rng=rng).record('v')
    sim.run(1000.0)
    data = outputs.get_data()
    sim.end()
    return data


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    scenario4(sim)
