from pyNN import space
import unittest
import numpy

def assert_arrays_almost_equal(a, b, threshold, msg=''):
    if a.shape != b.shape:
        raise unittest.TestCase.failureException("Shape mismatch: a.shape=%s, b.shape=%s" % (a.shape, b.shape))
    if not (abs(a-b) < threshold).all():
        err_msg = "%s != %s" % (a, b)
        err_msg += "\nlargest difference = %g" % abs(a-b).max()
        if msg:
            err_msg += "\nOther information: %s" % msg
        raise unittest.TestCase.failureException(err_msg)

class LineTest(unittest.TestCase):
    
    def test_generate_positions(self):
        line = space.Line()
        assert_arrays_almost_equal(
            line.generate_positions(3),
            numpy.array([[0,0,0], [1,0,0], [2,0,0]], float),
            threshold=1e-15
        )
        
        
class SphereTest(unittest.TestCase):
    
    def test__create(self):
        s = space.Sphere(1.0)
        self.assertEqual(s.radius, 1.0)
        
# ==============================================================================
if __name__ == "__main__":
    unittest.main()