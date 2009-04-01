from pyNN import utility
import unittest
import os

class ColouredOutputTests(unittest.TestCase):
    
    def test_colour(self):
        utility.colour(utility.red, "foo") # just check no Exceptions are raised
        

class NotifyTests(unittest.TestCase):
    
    def test_notify(self):
        utility.notify()
        
class GetArgTests(unittest.TestCase):
    
    def test_get_script_args(self):
        utility.get_script_args("utilitytests.py", 0)
        
    def test_get_script_args1(self):
        self.assertRaises(Exception, utility.get_script_args, "utilitytests.py", 1)
        
class InitLoggingTests(unittest.TestCase):
    
    def test_initlogging_debug(self):
        utility.init_logging("test.log", debug=True, num_processes=2, rank=99)
        assert os.path.exists("test.log.99")
        os.remove("test.log.99")


class MockSimA(object):
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
        self.ms = utility.MultiSim([MockSimA, MockSimB], MockNet, {})
    
    def tearDown(self):
        self.ms.end()
    
    def test_create(self):
        pass
    
    def test_getattr(self):  
        self.assertEqual(self.ms.f(), {'MockSimA': 2, 'MockSimB': 2})

    def test_run_simple(self):
        self.ms.run(100.0)
        
    def test_run_with_callback(self):
        self.ms.run(100.0, 2, lambda: 99)
        
    def test_iter(self):
        nets = [net for net in self.ms]
        self.assertEqual(nets, self.ms.nets.values())

import time

class TimerTest(unittest.TestCase):
    
    def test_timer(self):
        timer = utility.Timer()
        time.sleep(0.1)
        assert timer.elapsedTime() > 0
        assert isinstance(timer.elapsedTime(format='long'), basestring)
        timer.reset()

# ==============================================================================
if __name__ == "__main__":
    unittest.main()