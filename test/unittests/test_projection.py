"""
Tests of the common implementation of the Projection class, using the
pyNN.mock backend.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy
import os
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mock import Mock, patch
from .mocks import MockRNG
import pyNN.mock as sim

from pyNN import random, errors, space
from pyNN.parameters import Sequence

class ProjectionTest(unittest.TestCase):

    def setUp(self):
        sim.setup()
        self.p1 = sim.Population(7, sim.IF_cond_exp())
        self.p2 = sim.Population(4, sim.IF_cond_exp())
        self.p3 = sim.Population(5, sim.IF_curr_alpha())
        self.random_connect = sim.FixedNumberPostConnector(n=2, weights=0.123, delays=0.5)
        self.all2all = sim.AllToAllConnector(weights=0.456, delays=0.4)

    def test_create_simple(self):
        prj = sim.Projection(self.p1, self.p2, method=self.all2all)

    def test_create_with_presynaptic_assembly(self):
        prj = sim.Projection(self.p1 + self.p2, self.p2, method=self.all2all)

    def test_create_with_homogeneous_postsynaptic_assembly(self):
        prj = sim.Projection(self.p1, self.p1 + self.p2, method=self.all2all)

    def test_create_with_inhomogeneous_postsynaptic_assembly(self):
        self.assertRaises(errors.ConnectionError, sim.Projection,
                          self.p1, self.p1 + self.p3, method=self.all2all)

    def test_create_with_synapse_dynamics(self):
        prj = sim.Projection(self.p1, self.p2, method=self.all2all,
                             synapse_dynamics=sim.SynapseDynamics())

    def test_create_with_invalid_type(self):
        self.assertRaises(errors.ConnectionError, sim.Projection,
                          self.p1, "foo", method=self.all2all,
                          synapse_dynamics=sim.SynapseDynamics())

    def test_size_with_gather(self):
        prj = sim.Projection(self.p1, self.p2, method=self.all2all)
        self.assertEqual(prj.size(gather=True), self.p1.size * self.p2.size)

    #def test_set_weights(self):
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, method=Mock())
    #    prj.synapse_type = "foo"
    #    prj.post.local_cells = [0]
    #    prj.set = Mock()
    #    prj.setWeights(0.5)
    #    prj.set.assert_called_with('weight', 0.5)

    #def test_randomize_weights(self):
    #    orig_len = sim.Projection.__len__
    #    sim.Projection.__len__ = Mock(return_value=42)
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, method=Mock())
    #    prj.set = Mock()
    #    rd = Mock()
    #    rd.next = Mock(return_value=777)
    #    prj.randomizeWeights(rd)
    #    rd.next.assert_called_with(len(prj))
    #    prj.set.assert_called_with('weight', 777)
    #    sim.Projection.__len__ = orig_len
    #
    #def test_set_delays(self):
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, method=Mock())
    #    prj.set = Mock()
    #    prj.setDelays(0.5)
    #    prj.set.assert_called_with('delay', 0.5)
    #
    #def test_randomize_delays(self):
    #    orig_len = sim.Projection.__len__
    #    sim.Projection.__len__ = Mock(return_value=42)
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, method=Mock())
    #    prj.set = Mock()
    #    rd = Mock()
    #    rd.next = Mock(return_value=777)
    #    prj.randomizeDelays(rd)
    #    rd.next.assert_called_with(len(prj))
    #    prj.set.assert_called_with('delay', 777)
    #    sim.Projection.__len__ = orig_len
    #
    #def test_set_synapse_dynamics_param(self):
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, method=Mock())
    #    prj.set = Mock()
    #    prj.setSynapseDynamics('U', 0.5)
    #    prj.set.assert_called_with('U', 0.5)
    #
    def test_get_weights_as_list(self):
        prj = sim.Projection(self.p1, self.p2, method=self.all2all)
        weights = prj.get("weights", format="list")[:5]
        target = numpy.array(
            [(self.p1[0], self.p2[0], 0.456),
             (self.p1[1], self.p2[0], 0.456),
             (self.p1[2], self.p2[0], 0.456),
             (self.p1[3], self.p2[0], 0.456),
             (self.p1[4], self.p2[0], 0.456),])
        assert_array_equal(weights, target)

    def test_get_weights_as_list_no_address(self):
        prj = sim.Projection(self.p1, self.p2, method=self.all2all)
        weights = prj.get("weights", format="list", with_address=False)[:5]
        target = 0.456*numpy.ones((5,))
        assert_array_equal(weights, target)

    def test_get_weights_as_array(self):
        prj = sim.Projection(self.p1, self.p2, method=self.all2all)
        weights = prj.get("weights", format="array")
        target = 0.456*numpy.ones((self.p1.size, self.p2.size))
        assert_array_equal(weights, target)

    #def test_get_delays(self):
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, method=Mock())
    #    prj.get = Mock()
    #    prj.getDelays(format='list', gather=False)
    #    prj.get.assert_called_with('delay', 'list')

    def test_save_connections_with_gather(self):
        filename = "test.connections"
        if os.path.exists(filename):
            os.remove(filename)
        prj = sim.Projection(self.p1, self.p2, method=self.all2all)
        prj.save('connections', filename, gather=True)
        assert os.path.exists(filename)
        os.remove(filename)

    #def test_print_weights_as_list(self):
    #    filename = "test.weights"
    #    if os.path.exists(filename):
    #        os.remove(filename)
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, method=Mock())
    #    prj.get = Mock(return_value=range(5))
    #    prj.printWeights(filename, format='list', gather=False)
    #    prj.get.assert_called_with('weight', format='list', gather=False)
    #    assert os.path.exists(filename)
    #    os.remove(filename)
    #
    #def test_print_weights_as_array(self):
    #    filename = "test.weights"
    #    if os.path.exists(filename):
    #        os.remove(filename)
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, method=Mock())
    #    prj.get = Mock(return_value=numpy.arange(5.0))
    #    prj.printWeights(filename, format='array', gather=False)
    #    prj.get.assert_called_with('weight', format='array', gather=False)
    #    assert os.path.exists(filename)
    #    os.remove(filename)

    def test_describe(self):
        prj = sim.Projection(self.p1, self.p2, method=self.all2all,
                             synapse_dynamics=sim.SynapseDynamics())
        self.assertIsInstance(prj.describe(engine='string'), basestring)
        self.assertIsInstance(prj.describe(template=None), dict)

    def test_weightHistogram(self):
        prj = sim.Projection(self.p1, self.p2, method=self.all2all)
        n, bins = prj.weightHistogram(min=0.0, max=1.0)
        assert_array_equal(bins, numpy.linspace(0, 1.0, num=11))
        assert_array_equal(n, numpy.array([0, 0, 0, 0, prj.size(), 0, 0, 0, 0, 0]))
