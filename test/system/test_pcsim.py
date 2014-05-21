from nose.plugins.skip import SkipTest
from scenarios.registry import registry

try:
    import pyNN.pcsim
    have_pcsim = True
except ImportError:
    have_pcsim = False

def test_all():
    for scenario in registry:
        if "pcsim" not in scenario.exclude:
            scenario.description = scenario.__name__
            if have_pcsim:
                yield scenario, pyNN.pcsim
            else:
                raise SkipTest
            
            
def test_PoissonInputNeuron():
    if have_pcsim:
        import pypcsim as pcs
        import numpy
        net = pcs.SingleThreadNetwork()
        inputs1 = [net.create(pcs.PoissonInputNeuron(rate=20.0, duration=1e6)) for i in range(5)]
        inputs2 = [net.create(pcs.PoissonInputNeuron(rate=40.0, duration=1e6)) for i in range(5)]
        recorders = [net.object(net.record(input, pcs.SpikeTimeRecorder())) for input in inputs1+inputs2]
        net.simulate(10.0)
        spike_counts = numpy.array([recorder.spikeCount() for recorder in recorders])
        assert (100 < spike_counts[:5]).all()
        assert (300 > spike_counts[:5]).all()
        assert (300 < spike_counts[5:]).all()
        assert (500 > spike_counts[5:]).all()
    else:
        raise SkipTest
    
    