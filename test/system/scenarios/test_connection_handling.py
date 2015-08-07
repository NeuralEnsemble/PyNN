"""
Tests of the fine-grained connection API

"""

from __future__ import division
from nose.tools import assert_equal, assert_almost_equal, assert_is_instance
from numpy.testing import assert_array_equal
import numpy as np
from pyNN import common
from .registry import register


@register()
def connections_attribute(sim):
    sim.setup()
    p1 = sim.Population(4, sim.SpikeSourceArray())
    p2 = sim.Population(3, sim.IF_cond_exp())
    prj = sim.Projection(p1, p2, sim.FixedProbabilityConnector(0.7), sim.StaticSynapse(weight=0.05, delay=0.5))

    connections = list(prj.connections)
    assert_equal(len(connections), len(prj))
    assert_is_instance(connections[0], common.Connection)
connections_attribute.__test__ = False


@register()
def connection_access_weight_and_delay(sim):
    sim.setup()
    p1 = sim.Population(4, sim.SpikeSourceArray())
    p2 = sim.Population(3, sim.IF_cond_exp())
    prj = sim.Projection(p1, p2, sim.FixedProbabilityConnector(0.8), sim.StaticSynapse(weight=0.05, delay=0.5))

    connections = list(prj.connections)
    assert_almost_equal(connections[2].weight, 0.05, places=9)
    assert_almost_equal(connections[2].delay, 0.5, places=9)
    connections[2].weight = 0.0123
    connections[2].delay = 1.0
    target = np.empty((prj.size(), 2))
    target[:, 0] = 0.05
    target[:, 1] = 0.5
    target[2, 0] = 0.0123
    target[2, 1] = 1.0
    assert_array_equal(np.array(prj.get(('weight', 'delay'), format='list', with_address=False)),
                       target)
connection_access_weight_and_delay.__test__ = False



if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    connections_attribute(sim)
