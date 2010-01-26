from pyNN import multisim
import unittest
import os

class MockSimA(object):
    simulator = None
    @staticmethod
    def run(tstop): pass
    @staticmethod
    def end(): pass
    
class MockSimB(MockSimA):
    pass

class MockNet(object):
    def __init__(self, sim, parameters):
        pass
    def f(self):
        return 2

class MultiSimTests(unittest.TestCase):
    
    def setUp(self):
        self.ms = multisim.MultiSim([MockSimA, MockSimB], MockNet, {})
    
    def tearDown(self):
        self.ms.end()
    
    def test_create(self):
        pass
    
    def test_getattr(self):  
        self.assertEqual(self.ms.f(), {'MockSimA': 2, 'MockSimB': 2})

    def test_run_simple(self):
        self.ms.run(100.0)
        
    #def test_run_with_callback(self):
    #    self.ms.run(100.0, 2, lambda: 99)
        
    def test_iter(self):
        nets = [net for net in self.ms]
        self.assertEqual(nets, self.ms.processes.values())


# ==============================================================================
if __name__ == "__main__":
    unittest.main()