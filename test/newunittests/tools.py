import numpy

def assert_arrays_equal(a, b):
    assert isinstance(a, numpy.ndarray), "a is a %s" % type(a)
    assert isinstance(b, numpy.ndarray), "b is a %s" % type(b)
    assert a.shape == b.shape, "%s != %s" % (a,b)
    assert all(a.flatten()==b.flatten()), "%s != %s" % (a,b)
