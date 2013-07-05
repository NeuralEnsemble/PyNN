try:
    import pyNN.nest as sim
    import nest
except ImportError:
    nest = False
from pyNN.standardmodels import StandardCellType
try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy
from numpy.testing import assert_array_equal, assert_array_almost_equal


@unittest.skipUnless(nest, "Requires NEST")
class TestFunctions(unittest.TestCase):

    def tearDown(self):
        sim.setup(verbosity='error')

    def test_list_standard_models(self):
        cell_types = sim.list_standard_models()
        self.assertTrue(len(cell_types) > 10)
        self.assertIsInstance(cell_types[0], basestring)

    def test_setup(self):
        sim.setup(timestep=0.05, min_delay=0.1, max_delay=1.0,
                  verbosity='debug', spike_precision='off_grid',
                  recording_precision=4, threads=2, rng_seeds=[873465, 3487564])
        ks = nest.GetKernelStatus()
        self.assertEqual(ks['resolution'], 0.05)
        self.assertEqual(ks['local_num_threads'], 2)
        self.assertEqual(ks['rng_seeds'], [873465, 3487564])
        #self.assertEqual(ks['min_delay'], 0.1)
        #self.assertEqual(ks['max_delay'], 1.0)
        self.assertTrue(ks['off_grid_spiking'])

    def test_setup_with_rng_seeds(self):
        sim.setup(rng_seeds_seed=42, threads=3)
        self.assertEqual(len(nest.GetKernelStatus('rng_seeds')), 3)

    def test_run_0(self, ):  # see https://github.com/NeuralEnsemble/PyNN/issues/191
        sim.setup(timestep=0.123, min_delay=0.246)
        sim.run(0)
        self.assertEqual(sim.get_current_time(), 0.0)


@unittest.skipUnless(nest, "Requires NEST")
class TestPopulation(unittest.TestCase):

    def setUp(self):
        sim.setup()
        self.p = sim.Population(4, sim.IF_cond_exp(**{'tau_m': 12.3,
                                                      'cm': lambda i: 0.987 + 0.01*i,
                                                     'i_offset': numpy.array([-0.21, -0.20, -0.19, -0.18])}))

    def test_create_native(self):
        cell_type = sim.native_cell_type('iaf_neuron')
        p = sim.Population(3, cell_type())

    def test__get_parameters(self):
        ps = self.p._get_parameters('C_m', 'g_L', 'E_ex', 'I_e')
        ps.evaluate(simplify=True)
        assert_array_almost_equal(ps['C_m'], numpy.array([987, 997, 1007, 1017], float),
                                  decimal=12)
        assert_array_almost_equal(ps['I_e'], numpy.array([-210, -200, -190, -180], float),
                                  decimal=12)
        self.assertEqual(ps['E_ex'], 0.0)


@unittest.skipUnless(nest, "Requires NEST")
class TestProjection(unittest.TestCase):

    def setUp(self):
        sim.setup()
        self.p1 = sim.Population(7, sim.IF_cond_exp())
        self.p2 = sim.Population(4, sim.IF_cond_exp())
        self.p3 = sim.Population(5, sim.IF_curr_alpha())
        self.syn_rnd = sim.StaticSynapse(weight=0.123, delay=0.5)
        self.syn_a2a = sim.StaticSynapse(weight=0.456, delay=0.4)
        self.random_connect = sim.FixedNumberPostConnector(n=2)
        self.all2all = sim.AllToAllConnector()

    def test_create_simple(self):
        prj = sim.Projection(self.p1, self.p2, self.all2all, synapse_type=self.syn_a2a)

    def test_create_with_synapse_dynamics(self):
        prj = sim.Projection(self.p1, self.p2, self.all2all,
                             synapse_type=sim.TsodyksMarkramSynapse())


if __name__ == '__main__':
    unittest.main()