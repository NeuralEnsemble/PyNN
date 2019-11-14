"""
Tests of the common implementation of the simulation control functions, using
the pyNN.mock backend.

:copyright: Copyright 2006-2019 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
import pyNN.hardware.brainscales as sim    

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

if MPI:
    mpi_comm = MPI.COMM_WORLD

extra = {'loglevel': 0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
       

class TestSimulationControl(unittest.TestCase):

    def test_setup(self):
        self.assertRaises(Exception, sim.setup, min_delay=1.0, max_delay=0.9, **extra)
        self.assertRaises(Exception, sim.setup, mindelay=1.0, **extra)  # } common
        self.assertRaises(Exception, sim.setup, maxdelay=10.0, **extra)  # } misspellings
        self.assertRaises(Exception, sim.setup, dt=0.1, **extra)        # }
        self.assertRaises(Exception, sim.setup, timestep=0.1, min_delay=0.09, **extra)

    def test_end(self):
        sim.setup(**extra)
        sim.end()  # need a better test
    
    def test_run(self):
        sim.setup(**extra)
        self.assertEqual(sim.run(100.0), 100.0)
    
    def test_reset(self):
        sim.setup(**extra)
        sim.run(100.0)
        sim.reset()
        self.assertEqual(sim.get_current_time(), 0.0)
    
    def test_time_step(self):
        sim.setup(0.123, min_delay=0.246, **extra)
        self.assertEqual(sim.get_time_step(), 0.123)
    
    def test_min_delay(self):
        sim.setup(0.123, min_delay=0.246, **extra)
        self.assertEqual(sim.get_min_delay(), 0.246)
    
    def test_max_delay(self):
        sim.setup(max_delay=9.87, **extra)
        self.assertEqual(sim.get_max_delay(), 9.87)
    
    @unittest.skipUnless(MPI, "test requires mpi4py")
    def test_num_processes(self):
        self.assertEqual(sim.num_processes(), mpi_comm.size)
    
    @unittest.skipUnless(MPI, "test requires mpi4py")
    def test_rank(self):
        self.assertEqual(sim.rank(), mpi_comm.rank)


if __name__ == '__main__':
    unittest.main()
