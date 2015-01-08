"""
Tests of the common implementation of the Projection class, using the
pyNN.mock backend.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy
import os
import sys
from numpy.testing import assert_array_equal, assert_array_almost_equal

try:
    from unittest.mock import Mock, patch
except ImportError:
    from mock import Mock, patch
try:
    basestring
except NameError:
    basestring = str
from .mocks import MockRNG
import pyNN.mock as sim

from pyNN import random, errors, space
from pyNN.parameters import Sequence

from .backends.registry import register_class, register

def _sort_by_column(A, col):
    A = numpy.array(A)
    array_index = numpy.argsort(A[:, col], kind='mergesort')
    return A[array_index]

def setUp():
    pass

@register_class()
class ProjectionTest(unittest.TestCase):

    def setUp(self, sim=sim, **extra):
        sim.setup(**extra)
        self.p1 = sim.Population(7, sim.IF_cond_exp())
        self.p2 = sim.Population(4, sim.IF_cond_exp())
        self.p3 = sim.Population(5, sim.IF_curr_alpha())
        self.syn1 = sim.StaticSynapse(weight=0.006, delay=0.5)
        self.random_connect = sim.FixedNumberPostConnector(n=2)
        self.syn2 = sim.StaticSynapse(weight=0.012, delay=0.4)
        self.all2all = sim.AllToAllConnector()
        self.syn3 = sim.TsodyksMarkramSynapse(weight=0.012, delay=0.6, U=0.2, tau_rec=50)

    def tearDown(self, sim=sim):
        sim.end()
        
    @register()
    def test_create_simple(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)

    @register()
    def test_create_with_presynaptic_assembly(self, sim=sim):
        prj = sim.Projection(self.p1 + self.p2, self.p2, connector=self.all2all, synapse_type=self.syn2)

    @register()
    def test_create_with_homogeneous_postsynaptic_assembly(self, sim=sim):
        prj = sim.Projection(self.p1, self.p1 + self.p2, connector=self.all2all, synapse_type=self.syn2)

    @register(exclude=['hardware.brainscales'])
    def test_create_with_inhomogeneous_postsynaptic_assembly(self, sim=sim):
        self.assertRaises(errors.ConnectionError, sim.Projection,
                          self.p1, self.p1 + self.p3, connector=self.all2all, synapse_type=self.syn2)

    @register()
    def test_create_with_fast_synapse_dynamics(self, sim=sim):
        depressing = sim.TsodyksMarkramSynapse(U=0.5, tau_rec=80.0, tau_facil=0.0)
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all,
                             synapse_type=depressing)

    @register()
    def test_create_with_invalid_type(self, sim=sim):
        self.assertRaises(errors.ConnectionError, sim.Projection,
                          self.p1, "foo", connector=self.all2all,
                          synapse_type=self.syn2)

    @register()
    def test_size_with_gather(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
        self.assertEqual(prj.size(gather=True), self.p1.size * self.p2.size)

# Need to extend the mock backend before setting synaptic parameters can be properly tested

    #def test_set_weights(self, sim=sim):
    #    prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
    #    prj.set(weight=0.789)
    #    weights = prj.get("weight", format="array", gather=False)  # use gather False because we are faking the MPI
    #    target = 0.789*numpy.ones((self.p1.size, self.p2.size))
    #    assert_array_equal(weights, target)

    #def test_randomize_weights(self, sim=sim):
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
    #def test_set_delays(self, sim=sim):
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, connector=Mock())
    #    prj.set = Mock()
    #    prj.setDelays(0.5)
    #    prj.set.assert_called_with('delay', 0.5)
    #
    #def test_randomize_delays(self, sim=sim):
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
    #def test_set_synapse_dynamics_param(self, sim=sim):
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, connector=Mock())
    #    prj.set = Mock()
    #    prj.setComposedSynapseType('U', 0.5)
    #    prj.set.assert_called_with('U', 0.5)
    #
    @register()
    def test_get_weights_as_list(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
        weights = prj.get("weight", format="list")
        weights = _sort_by_column(weights, 1)[:5]
        target = numpy.array(
            [(0, 0, 0.012),
             (1, 0, 0.012),
             (2, 0, 0.012),
             (3, 0, 0.012),
             (4, 0, 0.012),])
        assert_array_equal(weights, target)

    @register()
    def test_get_weights_as_list_no_address(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
        weights = prj.get("weight", format="list", with_address=False)[:5]
        target = 0.012*numpy.ones((5,))
        assert_array_equal(weights, target)

    @register()
    def test_get_weights_as_array(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
        weights = prj.get("weight", format="array", gather=False)  # use gather False because we are faking the MPI
        target = 0.012*numpy.ones((self.p1.size, self.p2.size))
        assert_array_equal(weights, target)

    @register()
    def test_get_weights_as_array_with_multapses(self, sim=sim):
        C = sim.FixedNumberPreConnector(n=7, rng=MockRNG(delta=1))
        prj = sim.Projection(self.p2, self.p3, C, synapse_type=self.syn1)
        # because we use a fake RNG, it is always the last three presynaptic cells which receive the double connection
        target = numpy.array([
            [0.006, 0.006, 0.006, 0.006, 0.006],
            [0.012, 0.012, 0.012, 0.012, 0.012],
            [0.012, 0.012, 0.012, 0.012, 0.012],
            [0.012, 0.012, 0.012, 0.012, 0.012],
            ])
        weights = prj.get("weight", format="array", gather=False)  # use gather False because we are faking the MPI
        assert_array_equal(weights, target)

    @register()
    def test_synapse_with_lambda_parameter(self, sim=sim):
        syn = sim.StaticSynapse(weight=lambda d: 0.01+0.001*d)
        prj = sim.Projection(self.p1, self.p2, self.all2all, synapse_type=syn)

    @register()
    def test_parameter_StaticSynapse_random_distribution(self, sim=sim):
        weight = random.RandomDistribution('uniform', low=0.005, high=0.015, rng=MockRNG(start=0.01, delta=0.001))
        syn = sim.StaticSynapse(weight=weight)
        self.assertEqual(weight.next(), 0.01)
        
    @register()
    def test_parameter_TsodyksMarkramSynapse_random_distribution(self, sim=sim):
        U_distr = random.RandomDistribution('uniform', low=0.4, high=0.6, rng=MockRNG(start=0.5, delta=0.001))
        depressing = sim.TsodyksMarkramSynapse(U=U_distr, tau_rec=lambda d: 80.0+d, tau_facil=0.0)
        self.assertEqual(U_distr.next(), 0.5)
        
    @register()
    def test_get_plasticity_attribute_as_list(self, sim=sim):
        U_distr = random.RandomDistribution('uniform', low=0.4, high=0.6, rng=MockRNG(start=0.5, delta=0.001))
        depressing = sim.TsodyksMarkramSynapse(U=U_distr, tau_rec=lambda d: 80.0+d, tau_facil=0.0)
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
            [(0, 0, 80),
             (1, 0, 81),
             (2, 0, 82),
             (3, 0, 83),
             (4, 0, 84),])
        assert_array_equal(tau_rec, tau_rec_target)

    #def test_get_delays(self, sim=sim):
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, connector=Mock())
    #    prj.get = Mock()
    #    prj.getDelays(format='list', gather=False)
    #    prj.get.assert_called_with('delay', 'list')

    @register()
    def test_save_connections_with_gather(self, sim=sim):
        filename = "test.connections"
        if os.path.exists(filename):
            os.remove(filename)
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn3)
        prj.save('connections', filename, gather=True)
        assert os.path.exists(filename)
        os.remove(filename)

    #def test_print_weights_as_list(self, sim=sim):
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
    #def test_print_weights_as_array(self, sim=sim):
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

    @register()
    def test_describe(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all,
                             synapse_type=self.syn2)
        self.assertIsInstance(prj.describe(engine='string'), basestring)
        self.assertIsInstance(prj.describe(template=None), dict)

    @register()
    def test_weightHistogram(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all,
                             synapse_type=self.syn2)
        n, bins = prj.weightHistogram(min=0.0, max=0.05)
        assert_array_equal(bins, numpy.linspace(0, 0.05, num=11))
        assert_array_equal(n, numpy.array([0, 0, prj.size(), 0, 0, 0, 0, 0, 0, 0]))
