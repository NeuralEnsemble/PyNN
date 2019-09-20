"""
Definition of cell classes for the brian2 module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""


import numpy
import brian2
from pyNN.parameters import Sequence, simplify
from pyNN import errors
from pyNN.brian2 import simulator
import pdb

mV = brian2.mV
ms = brian2.ms
nA = brian2.nA
uS = brian2.uS
Hz = brian2.Hz
nF= brian2.nF
ampere = brian2.amp
second = brian2.second


def _new_property(obj_hierarchy, attr_name, units):
    """
    Return a new property, mapping attr_name to obj_hierarchy.attr_name.

    For example, suppose that an object of class A has an attribute b which
    itself has an attribute c which itself has an attribute d. Then placing
      e = _new_property('b.c', 'd')
    in the class definition of A makes A.e an alias for A.b.c.d
    """
    def set(self, value):
        if obj_hierarchy:
            obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        else:
            obj = self
        setattr(obj, attr_name, value * units)

    def get(self):
        if obj_hierarchy:
            obj = reduce(getattr, [self] + obj_hierarchy.split('.'))   ### ( getattr, ...)
        else:
            obj = self
        return getattr(obj, attr_name) / units
    return property(fset=set, fget=get)


class BaseNeuronGroup(brian2.NeuronGroup):
    
    def __init__(self, n, equations, threshold, reset=-20 * mV,
                 refractory=0 * ms, **parameters):
        if "tau_refrac" in parameters:
            max_refractory = parameters["tau_refrac"].max() * ms
        else:
            max_refractory = None 
        brian2.NeuronGroup.__init__(self, n,
                                   model=equations,
                                   threshold=threshold,
                                   reset=reset,
                                   refractory=refractory,
                                   clock=simulator.state.network.clock)
        for name, value in parameters.items():

            if not hasattr(self, name):
                self.add_attribute(name)
            if (name=="tau_refrac"):
                self.tau_refrac=value   
            else:    
                setattr(self, name, value)
        #self._S0 = self._S[:, 0]  # store parameter values in case of reset.
                                 # TODO: update this when parameters are modified
                                 # TODO: Brian2 does not have _S0a                        
        self.add_attribute('initial_values')
        self.initial_values = {}

    def initialize(self):
        for variable, values in self.initial_values.items():
            setattr(self, variable, values)


class ThresholdNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold = 'v > {}*mV'.format(parameters["v_thresh"][0]*1000) 
        self._resetvalue=parameters.pop('v_reset')[0] 
        self._refracvalue=parameters.pop('tau_refrac')[0]
        reset = 'v = {}*mV'.format(self._resetvalue*1000)
        refractory=self._refracvalue       
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset=reset,
                                 refractory=refractory, **parameters)
    @property 
    def v_reset (self):
        return self._resetvalue

    @property 
    def tau_refrac (self):  
        return self._refractory 

    @v_reset.setter
    def v_reset (self, resetvalue):
        self.event_codes['spike']= 'v = {}*mV'.format(resetvalue)

    @tau_refrac.setter
    def tau_refrac(self, tau_refrac_value): 
        self._refractory = tau_refrac_value * ms
          

    # def check_threshold(self, v):
    #     return v >= self.v_thresh


class BiophysicalNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold='v > -40*mV'
        refractory = 2 * ms
        reset = 'v=0*mV'
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold=threshold,reset=reset, refractory=refractory,
                                 **parameters)
                                 # implicit=True, compile=False,
    @property 
    def v_reset (self):
        return self._resetvalue
        #return self.event_codes['spike']
    @v_reset.setter
    def v_reset (self, resetvalue):
        #self._resetvalue = resetvalue * mV
        self.event_codes['spike']= 'v = {}*mV'.format(resetvalue)

class AdaptiveReset(object):

    def __init__(self, Vr=-70.7 * mV, b=0.0805 * nA):
        self.Vr = Vr
        self.b = b

    def __call__(self, P, spikes):
        P.v[spikes] = self.Vr[spikes]
        P.w[spikes] += self.b[spikes]


class AdaptiveNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        #threshold = 'v >= {}*mV'.format(parameters["v_thresh"][0]*1000)
        thresh= parameters["v_thresh"][0]
        Vcut= thresh + parameters["delta_T"][0]*5
        threshold = 'v > {}*mV'.format(Vcut*1000)
        self._resetvalue=parameters.pop('v_reset')[0] 
        self._bvalue=parameters.pop('b')[0] 
        #period = simplify(parameters['tau_refrac'])*1000 ##### problem here with the refractory
        self._refracvalue=parameters.pop('tau_refrac')[0] 
        reset = 'v = {}*mV; w+={}* amp'.format(self._resetvalue*10**3 ,self._bvalue)
        #refractory=None
        refractory= self._refracvalue
        #refractory=period
        #parameters['tau_refrac']=parameters['tau_refrac']/ms
       

        #refractory = 0*ms
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold=threshold, reset=reset, refractory=refractory,
                                 **parameters)


    @property 
    def v_reset (self):
        return self._resetvalue

    @property 
    def b (self):
        return self._bvalue
    @property 
    def tau_refrac (self):    
        return self._refractory    
        
    @tau_refrac.setter
    def tau_refrac(self, tau_refrac_value): 
        #self._refracvalue = tau_refrac_value * ms  
        #brian2.NeuronGroup.__setattr__(self, 'tau_refrac', tau_refrac_value)
        self._refractory = tau_refrac_value

    @v_reset.setter
    def v_reset (self, resetvalue):
        #self._resetvalue = resetvalue * mV
        self.event_codes['spike']= 'v = {}*mV'.format(resetvalue)
     

   


class AdaptiveReset2(object):
    def __init__(self, v_reset, q_r, q_s):
        self.v_reset = v_reset
        self.q_r = q_r
        self.q_s = q_s

    def __call__(self, P, spikes):
        P.v[spikes] = self.v_reset[spikes]
        P.g_r[spikes] += self.q_r[spikes]
        P.g_s[spikes] += self.q_s[spikes]


class AdaptiveNeuronGroup2(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold = 'v >= {}*mV'.format(parameters["v_thresh"][0]*1000)
        self._resetvalue=parameters.pop('v_reset')[0]
        self._q_rvalue=parameters.pop('q_r')[0] * 10**9
        self._q_svalue=parameters.pop('q_s')[0]* 10**9
        self._refracvalue=parameters.pop('tau_refrac')[0]
        reset = 'v = {}*mV; g_r+= {}*nS; g_s+={}*nS'.format(self._resetvalue*1000, self._q_rvalue, self._q_svalue)
        refractory=self._refracvalue
        '''
        threshold = brian2.SimpleFunThreshold(self.check_threshold)
        period = simplify(parameters['tau_refrac'])
        assert not hasattr(period, "__len__"), "Brian2 does not support heterogenerous refractory periods with CustomRefractoriness"
        reset = brian2.SimpleCustomRefractoriness(
                    AdaptiveReset2(parameters.pop('v_reset'),
                                   parameters.pop('q_r'),
                                   parameters.pop('q_s')),
                    period=period * second)
        refractory = None
        '''
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset=reset, refractory=refractory,
                                 **parameters)
    @property 
    def v_reset (self):
        return self._resetvalue
    @property 
    def q_r (self):
        return self._q_rvalue 
    @property 
    def q_s (self):
        return self._q_svalue  

    @property 
    def tau_refrac (self):
        return self._refractory 

    @tau_refrac.setter
    def tau_refrac(self, tau_refrac_value): 
        #self._refracvalue = tau_refrac_value * ms  
        #brian2.NeuronGroup.__setattr__(self, 'tau_refrac', tau_refrac_value)
        self._refractory = tau_refrac_value

    @v_reset.setter
    def v_reset (self, resetvalue):
        #self._resetvalue = resetvalue * mV
        self.event_codes['spike']= 'v = {}*mV'.format(resetvalue)

 
    '''
    tau_refrac = _new_property('', '_refractory_array', ms)
    v_reset = _new_property('_resetfun.resetfun', 'v_reset', mV)
    q_r = _new_property('_resetfun.resetfun', 'q_r', nA)
    q_s = _new_property('_resetfun.resetfun', 'q_s', nA)

    def check_threshold(self, v):
        return v >= self.v_thresh
    '''

class IzhikevichReset(object):

    def __init__(self, Vr=-65 * mV, d=0.2 * mV / ms):
        self.Vr = Vr
        self.d = d

    def __call__(self, P, spikes):
        P.v[spikes] = self.Vr[spikes]
        P.u[spikes] += self.d[spikes]


class IzhikevichNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        #threshold = brian2.SimpleFunThreshold(self.check_threshold)
        #threshold = 'v >= {}*mV'.format(parameters["v_thresh"][0]*1000)
        threshold='v >= 30*mV'
        self._resetvalue=parameters.pop('v_reset')[0]
        self._dvalue=parameters.pop('d')[0]
        reset = 'v = {}*mV; u+={}*mV/ms'.format(self._resetvalue*1000, self._dvalue)
        '''
        reset = brian2.SimpleCustomRefractoriness(
                    IzhikevichReset(parameters['v_reset'],
                                    parameters['d']),
                    period=0 * ms)
        '''            
        refractory = 0 * ms
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold=threshold, reset=reset, refractory=refractory,
                                 **parameters)
        #self._variable_refractory_time = True
        #self._refractory_variable = None
        #self._S0 = self._S[:, 0]

    #v_reset = _new_property('_resetfun.resetfun', 'Vr', mV)
    #b = _new_property('_resetfun.resetfun', 'b', nA)

    @property 
    def v_reset (self):
        return self._resetvalue
    @property 
    def d (self):
        return self._dvalue 

    @v_reset.setter
    def v_reset (self, resetvalue):
        #self._resetvalue = resetvalue * mV
        self.event_codes['spike']= 'v = {}*mV'.format(resetvalue)    

class PoissonGroup(brian2.PoissonGroup):

    def __init__(self, n, equations, **parameters):
        #pdb.set_trace()
        for name, value in parameters.items():
            setattr(self, name, value)
        rates = parameters["firing_rate"]   
        brian2.PoissonGroup.__init__(self, n,
                                    rates=rates,
                                    clock=simulator.state.network.clock)

    def update_rates(self, t):
        #print(t, self.rate)
        idx = (self.start <= t) & (t <= self.start + self.duration)
        return numpy.where(idx, self.firing_rate, 0)

    def initialize(self):
        pass

    def _get_rate(self):
        return self.firing_rate

    def _set_rate(self, value):
        self.firing_rate = value
    rate = property(fset=_set_rate, fget=_get_rate)


class SpikeGeneratorGroup(brian2.SpikeGeneratorGroup):

    def __init__(self, n, indices, times):
        """
        Note that `equations` is not used: it is simply for compatibility with
        other NeuronGroup subclasses.
        """
        #pdb.set_trace()
        #print(_neuron_index)
        #print(self._spike_time)
        self._check_spike_times(times)
        #self._get_spike_times()
        #indices= brian2.SpikeGeneratorGroup.neuron_index
        #times = brian2.SpikeGeneratorGroup.spike_time
        
        #spiketimes = [(i, t) for i, seq in enumerate(times) for t in seq.value] * second
        indices= numpy.array([])
       # times=numpy.array([])* second
        idx= [i for i, seq in enumerate(times)]
        times = [ t for t in seq.value] * ms
        brian2.SpikeGeneratorGroup.__init__(self, n, indices=indices, times=times)

    def _get_spike_times(self):
        values = [list() for i in range(self.N-1)]
        for i, t in self.spike_time:
            values[i].append(t)
        return numpy.array([Sequence(times) for times in values], dtype=Sequence)

    def _set_spike_times(self, times, mask=None):        
        if mask is not None:
            existing_times = self._get_spike_times()
            existing_times[mask] = times
            times = existing_times
        self._check_spike_times(times)
        values = [(i, t) for i, seq in enumerate(times) for t in seq.value]
        brian2.SpikeGeneratorGroup.__init__(self, self.N, values, period=self.period)
    times = property(fget=_get_spike_times, fset=_set_spike_times)

    def _check_spike_times(self, times):
        for seq in times:
            if numpy.any(seq.value[:-1] > seq.value[1:]):
                raise errors.InvalidParameterValueError(
                    "Spike times given to SpikeSourceArray must be in increasing order")

    def initialize(self):
        pass
