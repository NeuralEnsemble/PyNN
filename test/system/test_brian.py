from nose.plugins.skip import SkipTest
from scenarios import *

try:
    import pyNN.brian
    have_brian = True
except ImportError:
    have_brian = False

def test_all():
    if have_brian:
        scenario1(pyNN.brian)
    else:
        raise SkipTest