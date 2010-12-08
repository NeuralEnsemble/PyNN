from nose.plugins.skip import SkipTest
from scenarios import *

try:
    import pyNN.pcsim
    have_pcsim = True
except ImportError:
    have_pcsim = False

def test_all():
    for scenario in (scenario1, scenario2):
        if have_pcsim:
            yield scenario, pyNN.pcsim
        else:
            raise SkipTest