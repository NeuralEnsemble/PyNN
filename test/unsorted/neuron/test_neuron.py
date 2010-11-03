from neuron import h, nrn
from pyNN.neuron.cells import SingleCompartmentNeuron
import pyNN.neuron as sim

#class TestCell(nrn.Section):
#    conductance_based = True
#    parameter_names= []
#    
#    def __init__(self):
#        nrn.Section.__init__(self)
#        self.source = self(0.5)._ref_v
#        self.recording_time = 0
#        self.spike_times = h.Vector(0)
#        self.excitatory = h.ExpSyn(0.5, sec=self)
#        self.inhibitory = h.ExpSyn(0.5, sec=self)
#        self.v_init = -65.0
#        
#    def record_v(self, active):
#        if active:
#            self.vtrace = h.Vector()
#            self.vtrace.record(self(0.5)._ref_v)
#            if not self.recording_time:
#                self.record_times = h.Vector()
#                self.record_times.record(h._ref_t)
#                self.recording_time += 1
#        else:
#            self.vtrace = None
#            self.recording_time -= 1
#            if self.recording_time == 0:
#                self.record_times = None
#                
#    def record(self, active):
#        if active:
#            rec = h.NetCon(self.source, None, sec=self)
#            rec.record(self.spike_times)
#
#    def memb_init(self, v_init=None):
#        if v_init:
#            self.v_init = v_init
#        assert self.v_init is not None, "cell is a %s" % self.__class__.__name__
#        for seg in self:
#            seg.v = self.v_init 


class TestCell(SingleCompartmentNeuron):
    parameter_names = ["c_m", "i_offset", "v_init", "tau_e", "tau_i", "e_e", "e_i"]
    conductance_based = True
    
    def __init__(self, c_m, i_offset, v_init, tau_e, tau_i, e_e, e_i):
        SingleCompartmentNeuron.__init__(self, 'conductance', 'exp', c_m, i_offset,
                                         v_init, tau_e, tau_i, e_e, e_i)
        self.source = self.seg._ref_v

    def get_threshold(self):
        return 10.0
    
    def record(self, active):
        if active:
            rec = h.NetCon(self.source, None, sec=self)
            rec.record(self.spike_times)


cell_params = {
    "c_m": 1.0,
    "i_offset": 0.0,
    "v_init": -65.0,
    "tau_e": 2.0,
    "tau_i": 2.0,
    "e_e": 0.0,
    "e_i": -75.0,
}

sim.setup()

p0 = sim.Population(1, sim.SpikeSourcePoisson, {'rate': 100.0})
p1 = sim.Population(10, TestCell, cell_params)
p2 = sim.Population(10, sim.IF_cond_exp)

p1.record_v(1)
p1.record()
p2.record_v(1)

#curr = sim.DCSource()
#curr.inject_into(p1)

prj01 = sim.Projection(p0, p1, sim.AllToAllConnector())
prj12 = sim.Projection(p1, p2, sim.FixedProbabilityConnector(0.5))

prj01.setWeights(0.1)
prj12.setWeights(0.1)

sim.run(1000.0)

t,v = p1.get_v()[:, 1:3].T
#print p2.get_v()
import pylab
pylab.rcParams['interactive'] = True

pylab.plot(t, v)