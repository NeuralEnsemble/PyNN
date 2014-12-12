try:
    import pyNN.brian as sim
    import brian
except ImportError:
    brian = False
try:
    import unittest2 as unittest
except ImportError:
    import unittest
try:
    basestring
except NameError:
    basestring = str
import numpy
from numpy.testing import assert_array_equal, assert_array_almost_equal


class MockConnector(object):

    def connect(self, projection):
        pass


@unittest.skipUnless(brian, "Requires Brian")
class TestProjection(unittest.TestCase):

    def setUp(self):
        sim.setup()
        self.syn = sim.StaticSynapse(weight=0.123, delay=0.5)

    def test_partitioning(self):
        p1 = sim.Population(5, sim.IF_cond_exp())
        p2 = sim.Population(7, sim.IF_cond_exp())
        a = p1 + p2[1:4]
        # [0 2 3 4 5][x 1 2 3 x x x]
        prj = sim.Projection(a, a, MockConnector(), synapse_type=self.syn)
        presynaptic_indices = numpy.array([0, 3, 4, 6, 7])
        partitions = prj._partition(presynaptic_indices)
        self.assertEqual(len(partitions), 2)
        assert_array_equal(partitions[0], numpy.array([0, 3, 4]))
        assert_array_equal(partitions[1], numpy.array([2, 3]))

        # [0 1 2 3 4][x 1 2 3 x]
        self.assertEqual(prj._localize_index(0), (0, 0))
        self.assertEqual(prj._localize_index(3), (0, 3))
        self.assertEqual(prj._localize_index(5), (1, 1))
        self.assertEqual(prj._localize_index(7), (1, 3))

if __name__ == '__main__':
    unittest.main()
