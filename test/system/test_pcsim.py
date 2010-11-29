from nose.plugins.skip import SkipTest
from scenarios import *

try:
    import pyNN.pcsim
    have_pcsim = True
except ImportError:
    have_pcsim = False

def test_all():
    if have_pcsim:
        scenario1(pyNN.pcsim)
    else:
        raise SkipTest