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
        utility.get_script_args(0)
        
    def test_get_script_args1(self):
        self.assertRaises(Exception, utility.get_script_args, 1)
        
class InitLoggingTests(unittest.TestCase):
    
    def test_initlogging_debug(self):
        utility.init_logging("test.log", debug=True, num_processes=2, rank=99)
        assert os.path.exists("test.log.99")
        os.remove("test.log.99")

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