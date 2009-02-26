

import numpy

class STDPSynapse(object):


    def __init__(self, delay, presynaptic_spikes, postsynaptic_spikes,
                 w_init=0.01, w_min=0, w_max=0.1, A_plus=0.01, A_minus=0.01,
                 tau_plus=20.0, tau_minus=20.0, ddf=0):
        self.delay = delay
        self.w_min = w_min
        self.w_max = w_max
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_init = w_init
        psp_times = presynaptic_spikes + delay
        wa_pre_times = presynaptic_spikes + (1-ddf)*delay
        pre = wa_pre_times.reshape(len(wa_pre_times),1)
        pre = numpy.concatenate((pre, numpy.ones_like(pre)), axis=1)
        wa_post_times = postsynaptic_spikes + ddf*delay
        post = wa_post_times.reshape(len(wa_post_times),1)
        post = numpy.concatenate((post, -1*numpy.ones_like(post)), axis=1)
        events = numpy.concatenate((pre,post), axis=0)
        if events.size > 0:
            ordering = events[:,0].argsort()
            self.events = events[ordering]
        else:
            self.events = events
        self.reset()
    
    def reset(self):
        self.P = 0
        self.M = 0
        self.tlast_pre = 0
        self.tlast_post = 0
        self.deltaw = []
    
    def calc_weights(self, at_input_spiketimes=False):
        self.reset()
        update = {1: self.pre_synaptic_spike,
                  -1: self.post_synaptic_spike}
        for t, code in self.events:
            self.deltaw.append( update[code](t) )
        weights = self.w_init + numpy.add.accumulate(self.deltaw)
        if at_input_spiketimes:
            mask = self.events[:,1]==1
        else:
            mask = slice(None)
        return self.events[:,0][mask], weights[mask]
                
    def pre_synaptic_spike(self, t):
        self.P = self.P*numpy.exp((self.tlast_pre-t)/self.tau_plus) + self.A_plus
        interval = self.tlast_post - t
        assert interval < 0 
        self.tlast_pre = t
        deltaw = self.w_max * self.M * numpy.exp(interval/self.tau_minus)
        return deltaw
        
    def post_synaptic_spike(self, t):
        self.M = self.M*numpy.exp((self.tlast_post-t)/self.tau_minus) - self.A_minus
        interval = t - self.tlast_pre
        assert interval > 0 
        self.tlast_post = t
        deltaw = self.w_max * self.P * numpy.exp(-interval/self.tau_plus)
        return deltaw
        
        
def test():
    presynaptic_spikes = numpy.arange(5,105,10)
    postsynaptic_spikes = presynaptic_spikes + 2.5
    S = STDPSynapse(1.0, presynaptic_spikes, postsynaptic_spikes)
    return S