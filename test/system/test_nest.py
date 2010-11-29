from nose.plugins.skip import SkipTest
from scenarios import *

try:
    import pyNN.nest
    have_nest = True
except ImportError:
    have_nest = False

def test_all():
    if have_nest:
        scenario1(pyNN.nest)
    else:
        raise SkipTest