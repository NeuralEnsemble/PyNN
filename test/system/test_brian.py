from nose.plugins.skip import SkipTest
from scenarios import *

try:
    import pyNN.brian
    have_brian = True
except ImportError:
    have_brian = False

def test_all():
    for scenario in (scenario1, scenario2):
        if have_brian:
            yield scenario, pyNN.brian
        else:
            raise SkipTest