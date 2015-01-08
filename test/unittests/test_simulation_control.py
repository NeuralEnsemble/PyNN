"""
Tests of the common implementation of the simulation control functions, using
the pyNN.mock backend.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
import pyNN.mock as sim    

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from .backends.registry import register_class, register

if MPI:
    mpi_comm = MPI.COMM_WORLD

@register_class()
class TestSimulationControl(unittest.TestCase):
        
    def setUp(self, sim=sim, **extra):
        self.extra = {}
        self.extra.update(extra)
        pass
    
    def tearDown(self, sim=sim):
        pass
        
    @register()
    def test_setup(self, sim=sim):
        self.assertRaises(Exception, sim.setup, min_delay=1.0, max_delay=0.9, **self.extra)
        sim.end()
        self.assertRaises(Exception, sim.setup, mindelay=1.0, **self.extra)  # } common
        sim.end()
        self.assertRaises(Exception, sim.setup, maxdelay=10.0, **self.extra) # } misspellings
        sim.end()
        self.assertRaises(Exception, sim.setup, dt=0.1, **self.extra)        # }
        sim.end()
        self.assertRaises(Exception, sim.setup, timestep=0.1, min_delay=0.09, **self.extra)
        sim.end()

    @register()
    def test_end(self, sim=sim):
        sim.setup(**self.extra)
        sim.end() # need a better test
    
    @register()
    def test_run(self, sim=sim):
        sim.setup(**self.extra)
        self.assertAlmostEqual(sim.run(100.0), 100.0)
        sim.end()
        
    @register(exclude=['hardware.brainscales'])
    def test_run_twice(self, sim=sim):
        sim.setup(**self.extra)
        self.assertAlmostEqual(sim.run(100.0), 100.0)
        self.assertAlmostEqual(sim.run(100.0), 200.0)
        sim.end()
    
    @register()
    def test_reset(self, sim=sim):
        sim.setup(**self.extra)
        sim.run(100.0)
        sim.reset()
        self.assertEqual(sim.get_current_time(), 0.0)
        sim.end()
 
    @register()
    def test_current_time(self, sim=sim):
        sim.setup(timestep=0.1, **self.extra)
        sim.run(10.1)
        self.assertAlmostEqual(sim.get_current_time(), 10.1)
        sim.end()
        
    @register(exclude=['hardware.brainscales'])
    def test_current_time_two_runs(self, sim=sim):
        sim.setup(timestep=0.1, **self.extra)
        sim.run(10.1)
        self.assertAlmostEqual(sim.get_current_time(), 10.1)
        sim.run(23.4)
        self.assertAlmostEqual(sim.get_current_time(), 33.5)
        sim.end()
    
    @register()
    def test_time_step(self, sim=sim):
        sim.setup(0.123, min_delay=0.246, **self.extra)
        self.assertAlmostEqual(sim.get_time_step(), 0.123)
        sim.end()
    
    @register()
    def test_min_delay(self, sim=sim):
        sim.setup(0.123, min_delay=0.246, **self.extra)
        self.assertEqual(sim.get_min_delay(), 0.246)
        sim.end()
    
    @register()
    def test_max_delay(self, sim=sim):
        sim.setup(max_delay=9.87, **self.extra)
        self.assertAlmostEqual(sim.get_max_delay(), 9.87)
        sim.end()

    @register(exclude=['hardware.brainscales'])
    def test_callbacks(self, sim=sim):
        total_time = 100.
        callback_steps = [10., 10., 20., 25.]

        # callbacks are called at 0. and after every step
        expected_callcount = [11, 11, 6, 5]
        num_callbacks = len(callback_steps)
        callback_callcount = [0] * num_callbacks

        def make_callback(idx):
            def callback(time):
                callback_callcount[idx] += 1
                return time + callback_steps[idx]
            return callback

        callbacks = [make_callback(i) for i in range(num_callbacks)]
        sim.setup(timestep=0.1, min_delay=0.1, **self.extra)
        sim.run_until(total_time, callbacks=callbacks)

        self.assertTrue(all(callback_callcount[i] == expected_callcount[i]
            for i in range(num_callbacks)))
        
        sim.end()
    
    @unittest.skipUnless(MPI, "test requires mpi4py")
    def test_num_processes(self, sim=sim):
        self.assertEqual(sim.num_processes(), mpi_comm.size)
    
    @unittest.skipUnless(MPI, "test requires mpi4py")
    def test_rank(self, sim=sim):
        self.assertEqual(sim.rank(), mpi_comm.rank)


if __name__ == '__main__':
    unittest.main()
