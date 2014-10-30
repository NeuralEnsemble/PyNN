"""
min_delay should be calculated automatically if not set
"""

from nose.tools import assert_equal
from .registry import register


@register(exclude="neuron")  # for NEURON, this only works when run with MPI and more than one process
def issue231(sim):
    sim.setup(min_delay='auto')

    p1 = sim.Population(13, sim.IF_cond_exp())
    p2 = sim.Population(25, sim.IF_cond_exp())

    connector = sim.AllToAllConnector()
    synapse = sim.StaticSynapse(delay=0.5)
    prj = sim.Projection(p1, p2, connector, synapse)
    sim.run(100.0)
    assert_equal(sim.get_min_delay(), 0.5)
    sim.end()


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    issue231(sim)