from pyNN import utility
try:
    import unittest2 as unittest
except ImportError:
    import unittest
import os
import time
import sys
try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO
try:
    basestring
except NameError:
    basestring = str


class NotifyTests(unittest.TestCase):
    
    def test_notify(self):
        utility.notify()
        
class GetArgTests(unittest.TestCase):
    
    def test_get_script_args(self):
        utility.get_script_args(0)

# fails with nose, passes with python       
#    def test_get_script_args1(self):
#        self.assertRaises(Exception, utility.get_script_args, 1)
 
# fails with nose, passes with python       
#class InitLoggingTests(unittest.TestCase):
#    
#    def test_initlogging_debug(self):
#        utility.init_logging("test.log", debug=True, num_processes=2, rank=99)
#        assert os.path.exists("test.log.99")
#        os.remove("test.log.99")


class TimerTest(unittest.TestCase):
    
    def test_timer(self):
        timer = utility.Timer()
        time.sleep(0.1)
        assert timer.elapsed_time() > 0
        assert isinstance(timer.elapsedTime(format='long'), basestring)
        timer.reset()

    def test_diff(self):
        timer = utility.Timer()
        time.sleep(0.1)
        self.assertAlmostEqual(timer.diff(), 0.1, places=2)
        time.sleep(0.2)
        self.assertAlmostEqual(timer.diff(), 0.2, places=1)
        self.assertAlmostEqual(timer.elapsed_time(), 0.3, places=2)


class ProgressBarTest(unittest.TestCase):
    
    def test_fixed(self):
        bar = utility.ProgressBar(width=12, mode='fixed')
        sys.stdout = StringIO()
        bar(0)
        bar(0.5)
        bar(1)
        sys.stdout.seek(0)
        self.assertEqual(sys.stdout.read(),
                         "[            ]   0% \r"
                         "[ #####      ]  50% \r"
                         "[ ########## ] 100% \r")
        sys.stdout = sys.__stdout__

    def test_dynamic(self):
        bar = utility.ProgressBar(width=12, mode='dynamic')
        sys.stdout = StringIO()
        bar(0)
        bar(0.5)
        bar(1)
        sys.stdout.seek(0)
        self.assertEqual(sys.stdout.read(),
                         "[  ]   0% \r"
                         "[ ##### ]  50% \r"
                         "[ ########## ] 100% \r")
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    unittest.main()

