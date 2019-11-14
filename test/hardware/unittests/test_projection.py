"""
Tests of the common implementation of the Projection class, using the
pyNN.mock backend.

:copyright: Copyright 2006-2019 by the PyNN team, see AUTHORS.
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
#import pyNN.neuron as sim
#import pyNN.nest as sim

from pyNN import random, errors, space
from pyNN.parameters import Sequence


def _sort_by_column(A, col):
    A = numpy.array(A)
    array_index = numpy.argsort(A[:, col], kind='mergesort')
    return A[array_index]


class ProjectionTest(unittest.TestCase):

    def setUp(self):
        sim.setup()
        self.p1 = sim.Population(7, sim.IF_cond_exp())
        self.p2 = sim.Population(4, sim.IF_cond_exp())
        self.p3 = sim.Population(5, sim.IF_curr_alpha())
        self.syn1 = sim.StaticSynapse(weight=0.123, delay=0.5)
        self.random_connect = sim.FixedNumberPostConnector(n=2)
        self.syn2 = sim.StaticSynapse(weight=0.456, delay=0.4)
        self.all2all = sim.AllToAllConnector()
        self.syn3 = sim.TsodyksMarkramSynapse(weight=0.789, delay=0.6, U=100.0, tau_rec=500)

    def test_create_simple(self):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)

    def test_create_with_presynaptic_assembly(self):
        prj = sim.Projection(self.p1 + self.p2, self.p2, connector=self.all2all, synapse_type=self.syn2)

    def test_create_with_homogeneous_postsynaptic_assembly(self):
        prj = sim.Projection(self.p1, self.p1 + self.p2, connector=self.all2all, synapse_type=self.syn2)

    def test_create_with_inhomogeneous_postsynaptic_assembly(self):
        self.assertRaises(errors.ConnectionError, sim.Projection,
                          self.p1, self.p1 + self.p3, connector=self.all2all, synapse_type=self.syn2)

    def test_create_with_fast_synapse_dynamics(self):
        depressing = sim.TsodyksMarkramSynapse(U=0.5, tau_rec=800.0, tau_facil=0.0)
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all,
                             synapse_type=depressing)

    def test_create_with_invalid_type(self):
        self.assertRaises(errors.ConnectionError, sim.Projection,
                          self.p1, "foo", connector=self.all2all,
                          synapse_type=self.syn2)

    def test_size_with_gather(self):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
        self.assertEqual(prj.size(gather=True), self.p1.size * self.p2.size)

# Need to extend the mock backend before setting synaptic parameters can be properly tested

    #def test_set_weights(self):
    #    prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
    #    prj.set(weight=0.789)
    #    weights = prj.get("weight", format="array", gather=False)  # use gather False because we are faking the MPI
    #    target = 0.789*numpy.ones((self.p1.size, self.p2.size))
    #    assert_array_equal(weights, target)

    #def test_randomize_weights(self):
    #    orig_len = sim.Projection.__len__
    #    sim.Projection.__len__ = Mock(return_value=42)
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, connector=Mock())
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
    #    prj = sim.Projection(p1, p2, connector=Mock())
    #    prj.set = Mock()
    #    prj.setDelays(0.5)
    #    prj.set.assert_called_with('delay', 0.5)
    #
    #def test_randomize_delays(self):
    #    orig_len = sim.Projection.__len__
    #    sim.Projection.__len__ = Mock(return_value=42)
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, connector=Mock())
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
    #    prj = sim.Projection(p1, p2, connector=Mock())
    #    prj.set = Mock()
    #    prj.setComposedSynapseType('U', 0.5)
    #    prj.set.assert_called_with('U', 0.5)
    #
    def test_get_weights_as_list(self):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
        weights = prj.get("weight", format="list")
        weights = _sort_by_column(weights, 1)[:5]
        target = numpy.array(
            [(0, 0, 0.456),
             (1, 0, 0.456),
             (2, 0, 0.456),
             (3, 0, 0.456),
             (4, 0, 0.456),])
        assert_array_equal(weights, target)

    def test_get_weights_as_list_no_address(self):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
        weights = prj.get("weight", format="list", with_address=False)[:5]
        target = 0.456 * numpy.ones((5,))
        assert_array_equal(weights, target)

    def test_get_weights_as_array(self):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
        weights = prj.get("weight", format="array", gather=False)  # use gather False because we are faking the MPI
        target = 0.456 * numpy.ones((self.p1.size, self.p2.size))
        assert_array_equal(weights, target)

    def test_get_weights_as_array_with_multapses(self):
        C = sim.FixedNumberPreConnector(n=7, rng=MockRNG(delta=1))
        prj = sim.Projection(self.p2, self.p3, C, synapse_type=self.syn1)
        # because we use a fake RNG, it is always the last three presynaptic cells which receive the double connection
        target = numpy.array([
            [0.123, 0.123, 0.123, 0.123, 0.123],
            [0.246, 0.246, 0.246, 0.246, 0.246],
            [0.246, 0.246, 0.246, 0.246, 0.246],
            [0.246, 0.246, 0.246, 0.246, 0.246],
            ])
        weights = prj.get("weight", format="array", gather=False)  # use gather False because we are faking the MPI
        assert_array_equal(weights, target)

    def test_get_plasticity_attribute_as_list(self):
        U_distr = random.RandomDistribution('uniform', low=0.4, high=0.6, rng=MockRNG(start=0.5, delta=0.001))
        depressing = sim.TsodyksMarkramSynapse(U=U_distr, tau_rec=lambda d: 800.0 + d, tau_facil=0.0)
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all,
                             synapse_type=depressing)
        U = prj.get("U", format="list")
        U = _sort_by_column(U, 1)[:5]
        U_target = numpy.array(
            [(0, 0, 0.5),
             (1, 0, 0.501),
             (2, 0, 0.502),
             (3, 0, 0.503),
             (4, 0, 0.504),])
        assert_array_equal(U, U_target)
        tau_rec = prj.get("tau_rec", format="list")
        tau_rec = _sort_by_column(tau_rec, 1)[:5]
        tau_rec_target = numpy.array(
            [(0, 0, 800),
             (1, 0, 801),
             (2, 0, 802),
             (3, 0, 803),
             (4, 0, 804),])
        assert_array_equal(tau_rec, tau_rec_target)

    #def test_get_delays(self):
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, connector=Mock())
    #    prj.get = Mock()
    #    prj.getDelays(format='list', gather=False)
    #    prj.get.assert_called_with('delay', 'list')

    def test_save_connections_with_gather(self):
        filename = "test.connections"
        if os.path.exists(filename):
            os.remove(filename)
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn3)
        prj.save('connections', filename, gather=True)
        assert os.path.exists(filename)
        os.remove(filename)

    #def test_print_weights_as_list(self):
    #    filename = "test.weights"
    #    if os.path.exists(filename):
    #        os.remove(filename)
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, connector=Mock())
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
    #    prj = sim.Projection(p1, p2, connector=Mock())
    #    prj.get = Mock(return_value=numpy.arange(5.0))
    #    prj.printWeights(filename, format='array', gather=False)
    #    prj.get.assert_called_with('weight', format='array', gather=False)
    #    assert os.path.exists(filename)
    #    os.remove(filename)

    def test_describe(self):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all,
                             synapse_type=self.syn2)
        self.assertIsInstance(prj.describe(engine='string'), basestring)
        self.assertIsInstance(prj.describe(template=None), dict)

    def test_weightHistogram(self):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all,
                             synapse_type=self.syn2)
        n, bins = prj.weightHistogram(min=0.0, max=1.0)
        assert_array_equal(bins, numpy.linspace(0, 1.0, num=11))
        assert_array_equal(n, numpy.array([0, 0, 0, 0, prj.size(), 0, 0, 0, 0, 0]))
