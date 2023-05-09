"""
Tests of the common implementation of the Projection class, using the
pyNN.mock backend.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import unittest
import numpy as np
import os
import sys
from numpy.testing import assert_array_equal

try:
    from unittest.mock import Mock, patch
except ImportError:
    from mock import Mock, patch
from .mocks import MockRNG
import pyNN.mock as sim

from pyNN import random, errors, space, standardmodels
from pyNN.parameters import Sequence


def _sort_by_column(A, col):
    A = np.array(A)
    array_index = np.argsort(A[:, col], kind='mergesort')
    return A[array_index]


def setUp():
    pass


class ProjectionTest(unittest.TestCase):

    def setUp(self, sim=sim, **extra):
        sim.setup(**extra)
        self.p1 = sim.Population(7, sim.IF_cond_exp())
        self.p2 = sim.Population(4, sim.IF_cond_exp())
        self.p3 = sim.Population(5, sim.IF_curr_alpha())
        self.syn1 = sim.StaticSynapse(weight=0.006, delay=0.5)
        self.random_connect = sim.FixedNumberPostConnector(n=2)
        self.syn2 = sim.StaticSynapse(weight=0.007, delay=0.4)
        self.all2all = sim.AllToAllConnector()
        self.syn3 = sim.TsodyksMarkramSynapse(weight=0.012, delay=0.6, U=0.2, tau_rec=50)

    def tearDown(self, sim=sim):
        sim.end()

    def test_create_simple(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)

    def test_create_with_presynaptic_assembly(self, sim=sim):
        prj = sim.Projection(self.p1 + self.p2, self.p2,
                             connector=self.all2all, synapse_type=self.syn2)

    def test_create_with_homogeneous_postsynaptic_assembly(self, sim=sim):
        prj = sim.Projection(self.p1, self.p1 + self.p2,
                             connector=self.all2all, synapse_type=self.syn2)

    def test_create_with_inhomogeneous_postsynaptic_assembly(self, sim=sim):
        self.assertRaises(errors.ConnectionError, sim.Projection,
                          self.p1, self.p1 + self.p3, connector=self.all2all, synapse_type=self.syn2)

    def test_create_with_fast_synapse_dynamics(self, sim=sim):
        depressing = sim.TsodyksMarkramSynapse(U=0.5, tau_rec=80.0, tau_facil=0.0)
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all,
                             synapse_type=depressing)

    def test_create_with_invalid_type(self, sim=sim):
        self.assertRaises(errors.ConnectionError, sim.Projection,
                          self.p1, "foo", connector=self.all2all,
                          synapse_type=self.syn2)

    def test_create_with_default_receptor_type(self, sim=sim):
        prj = sim.Projection(self.p1, self.p3, connector=self.all2all,
                             synapse_type=sim.StaticSynapse())
        self.assertEqual(prj.receptor_type, "excitatory")
        prj = sim.Projection(self.p1, self.p3, connector=self.all2all,
                             synapse_type=sim.TsodyksMarkramSynapse(weight=0.5))
        self.assertEqual(prj.receptor_type, "excitatory")
        prj = sim.Projection(self.p1, self.p3, connector=self.all2all,
                             synapse_type=sim.StaticSynapse(weight=lambda d: -0.1 * d))
        self.assertEqual(prj.receptor_type, "inhibitory")

    def test_size_with_gather(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
        self.assertEqual(prj.size(gather=True), self.p1.size * self.p2.size)

# Need to extend the mock backend before setting synaptic parameters can be properly tested

    # def test_set_weights(self, sim=sim):
    #    prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
    #    prj.set(weight=0.789)
    #    weights = prj.get("weight", format="array", gather=False)  # use gather False because we are faking the MPI
    #    target = 0.789*np.ones((self.p1.size, self.p2.size))
    #    assert_array_equal(weights, target)

    # def test_randomize_weights(self, sim=sim):
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
    # def test_set_delays(self, sim=sim):
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, connector=Mock())
    #    prj.set = Mock()
    #    prj.setDelays(0.5)
    #    prj.set.assert_called_with('delay', 0.5)
    #
    # def test_randomize_delays(self, sim=sim):
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
    # def test_set_synapse_dynamics_param(self, sim=sim):
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, connector=Mock())
    #    prj.set = Mock()
    #    prj.setComposedSynapseType('U', 0.5)
    #    prj.set.assert_called_with('U', 0.5)
    #

    def test_get_weights_as_list(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
        weights = prj.get("weight", format="list")
        weights = _sort_by_column(weights, 1)[:5]
        target = np.array(
            [(0, 0, 0.007),
             (1, 0, 0.007),
             (2, 0, 0.007),
             (3, 0, 0.007),
             (4, 0, 0.007), ])
        assert_array_equal(weights, target)

    def test_get_weights_as_list_no_address(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
        weights = prj.get("weight", format="list", with_address=False)[:5]
        target = 0.007 * np.ones((5,))
        assert_array_equal(weights, target)

    def test_get_weights_as_array(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn2)
        # use gather False because we are faking the MPI
        weights = prj.get("weight", format="array", gather=False)
        target = 0.007 * np.ones((self.p1.size, self.p2.size))
        assert_array_equal(weights, target)

    def test_get_weights_as_array_with_multapses(self, sim=sim):
        C = sim.FixedNumberPreConnector(n=7, rng=MockRNG(delta=1))
        prj = sim.Projection(self.p2, self.p3, C, synapse_type=self.syn1)
        # because we use a fake RNG, it is always the last three presynaptic cells which receive the double connection
        target = np.array([
            [0.006, 0.006, 0.006, 0.006, 0.006],
            [0.012, 0.012, 0.012, 0.012, 0.012],
            [0.012, 0.012, 0.012, 0.012, 0.012],
            [0.012, 0.012, 0.012, 0.012, 0.012],
        ])
        # use gather False because we are faking the MPI
        weights = prj.get("weight", format="array", gather=False)
        assert_array_equal(weights, target)

    def test_get_weights_as_array_with_multapses_min(self, sim=sim):
        C = sim.FixedNumberPreConnector(n=7, rng=MockRNG(delta=1))
        prj = sim.Projection(self.p2, self.p3, C, synapse_type=self.syn1)
        target = np.array([
            [0.006, 0.006, 0.006, 0.006, 0.006],
            [0.006, 0.006, 0.006, 0.006, 0.006],
            [0.006, 0.006, 0.006, 0.006, 0.006],
            [0.006, 0.006, 0.006, 0.006, 0.006],
        ])
        # use gather False because we are faking the MPI
        weights = prj.get("weight", format="array", gather=False, multiple_synapses='min')
        assert_array_equal(weights, target)

    def test_synapse_with_lambda_parameter(self, sim=sim):
        syn = sim.StaticSynapse(weight=lambda d: 0.01 + 0.001 * d)
        prj = sim.Projection(self.p1, self.p2, self.all2all, synapse_type=syn)

    def test_parameter_StaticSynapse_random_distribution(self, sim=sim):
        weight = random.RandomDistribution(
            'uniform', low=0.005, high=0.015, rng=MockRNG(start=0.01, delta=0.001))
        syn = sim.StaticSynapse(weight=weight)
        self.assertEqual(weight.next(), 0.01)

    def test_parameter_TsodyksMarkramSynapse_random_distribution(self, sim=sim):
        U_distr = random.RandomDistribution(
            'uniform', low=0.4, high=0.6, rng=MockRNG(start=0.5, delta=0.001))
        depressing = sim.TsodyksMarkramSynapse(
            U=U_distr, tau_rec=lambda d: 80.0 + d, tau_facil=0.0)
        self.assertEqual(U_distr.next(), 0.5)

    def test_get_plasticity_attribute_as_list(self, sim=sim):
        U_distr = random.RandomDistribution(
            'uniform', low=0.4, high=0.6, rng=MockRNG(start=0.5, delta=0.001))
        depressing = sim.TsodyksMarkramSynapse(
            U=U_distr, tau_rec=lambda d: 80.0 + d, tau_facil=0.0)
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all,
                             synapse_type=depressing)
        U = prj.get("U", format="list")
        U = _sort_by_column(U, 1)[:5]
        U_target = np.array(
            [(0, 0, 0.5),
             (1, 0, 0.501),
             (2, 0, 0.502),
             (3, 0, 0.503),
             (4, 0, 0.504), ])
        assert_array_equal(U, U_target)
        tau_rec = prj.get("tau_rec", format="list")
        tau_rec = _sort_by_column(tau_rec, 1)[:5]
        tau_rec_target = np.array(
            [(0, 0, 80),
             (1, 0, 81),
             (2, 0, 82),
             (3, 0, 83),
             (4, 0, 84), ])
        assert_array_equal(tau_rec, tau_rec_target)

    # def test_get_delays(self, sim=sim):
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, connector=Mock())
    #    prj.get = Mock()
    #    prj.getDelays(format='list', gather=False)
    #    prj.get.assert_called_with('delay', 'list')

    def test_save_connections_with_gather(self, sim=sim):
        filename = "test.connections"
        if os.path.exists(filename):
            os.remove(filename)
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all, synapse_type=self.syn3)
        prj.save('connections', filename, gather=True)
        assert os.path.exists(filename)
        os.remove(filename)

    # def test_print_weights_as_list(self, sim=sim):
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
    # def test_print_weights_as_array(self, sim=sim):
    #    filename = "test.weights"
    #    if os.path.exists(filename):
    #        os.remove(filename)
    #    p1 = sim.Population(7, sim.IF_cond_exp)
    #    p2 = sim.Population(7, sim.IF_cond_exp)
    #    prj = sim.Projection(p1, p2, connector=Mock())
    #    prj.get = Mock(return_value=np.arange(5.0))
    #    prj.printWeights(filename, format='array', gather=False)
    #    prj.get.assert_called_with('weight', format='array', gather=False)
    #    assert os.path.exists(filename)
    #    os.remove(filename)

    def test_describe(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all,
                             synapse_type=self.syn2)
        self.assertIsInstance(prj.describe(engine='string'), str)
        self.assertIsInstance(prj.describe(template=None), dict)

    def test_weightHistogram(self, sim=sim):
        prj = sim.Projection(self.p1, self.p2, connector=self.all2all,
                             synapse_type=self.syn2)
        n, bins = prj.weightHistogram(min=0.0, max=0.05)
        assert_array_equal(bins, np.linspace(0, 0.05, num=11))
        assert_array_equal(n, np.array([0, prj.size(), 0, 0, 0, 0, 0, 0, 0, 0]))


class CheckTest(unittest.TestCase):

    def setUp(self, sim=sim, **extra):
        self.MIN_DELAY = 0.123
        sim.setup(num_processes=2, rank=1, min_delay=self.MIN_DELAY, **extra)
        self.p1 = sim.Population(7, sim.IF_cond_exp())
        self.p2 = sim.Population(4, sim.IF_cond_exp())
        self.p3 = sim.Population(5, sim.IF_curr_alpha())
        self.projections = {
            "cond": {},
            "curr": {}
        }
        for psr, post in (("cond", self.p2), ("curr", self.p3)):
            for rt in ("excitatory", "inhibitory"):
                self.projections[psr][rt] = sim.Projection(
                    self.p1, post, sim.AllToAllConnector(safe=True),
                    sim.StaticSynapse(), receptor_type=rt
                )

    def test_check_weights_with_scalar(self, sim=sim):
        # positive weight
        for prj in [
            self.projections["cond"]["excitatory"],
            self.projections["curr"]["excitatory"],
            self.projections["cond"]["inhibitory"],
        ]:
            standardmodels.check_weights(4.3, prj)
        for prj in [
            self.projections["curr"]["inhibitory"],
        ]:
            self.assertRaises(errors.ConnectionError, standardmodels.check_weights, 4.3, prj)

        # negative weight
        for prj in [
            self.projections["cond"]["excitatory"],
            self.projections["curr"]["excitatory"],
            self.projections["cond"]["inhibitory"],
        ]:
            self.assertRaises(errors.ConnectionError, standardmodels.check_weights, -4.3, prj)
        for prj in [
            self.projections["curr"]["inhibitory"],
        ]:
            standardmodels.check_weights(-4.3, prj)

    def test_check_weights_with_array(self, sim=sim):
        # all positive weights
        w = np.arange(10)
        for prj in [
            self.projections["cond"]["excitatory"],
            self.projections["curr"]["excitatory"],
            self.projections["cond"]["inhibitory"],
        ]:
            standardmodels.check_weights(w, prj)
        for prj in [
            self.projections["curr"]["inhibitory"],
        ]:
            self.assertRaises(errors.ConnectionError, standardmodels.check_weights, w, prj)
        # all negative weights
        w = np.arange(-10, 0)
        for prj in [
            self.projections["cond"]["excitatory"],
            self.projections["curr"]["excitatory"],
            self.projections["cond"]["inhibitory"],
        ]:
            self.assertRaises(errors.ConnectionError, standardmodels.check_weights, w, prj)
        for prj in [
            self.projections["curr"]["inhibitory"],
        ]:
            standardmodels.check_weights(w, prj)
        # mixture of positive and negative weights
        w = np.arange(-5, 5)
        for prj in [
            self.projections["cond"]["excitatory"],
            self.projections["curr"]["excitatory"],
            self.projections["cond"]["inhibitory"],
            self.projections["curr"]["inhibitory"]
        ]:
            self.assertRaises(errors.ConnectionError, standardmodels.check_weights, w, prj)

    def test_check_weights_with_invalid_value(self, sim=sim):
        w = "butterflies"
        for prj in [
            self.projections["cond"]["excitatory"],
            self.projections["curr"]["excitatory"],
            self.projections["cond"]["inhibitory"],
            self.projections["curr"]["inhibitory"]
        ]:
            self.assertRaises(errors.ConnectionError, standardmodels.check_weights, w, prj)
