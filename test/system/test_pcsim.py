from nose.plugins.skip import SkipTest
from scenarios import scenarios

try:
    import pyNN.pcsim
    have_pcsim = True
except ImportError:
    have_pcsim = False

def test_all():
    for scenario in scenarios:
        scenario.description = scenario.__name__
        if have_pcsim:
            yield scenario, pyNN.pcsim
        else:
            raise SkipTest