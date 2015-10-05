# encoding: utf-8
"""
Definition of cell classes for the neuron module.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN.models import BaseCellType
from pyNN import errors
from neuron import h, nrn, hclass
from .simulator import state
from math import pi
import logging
from .recording import recordable_pattern
try:
    reduce
except NameError:
    from functools import reduce

logger = logging.getLogger("PyNN")


def _new_property(obj_hierarchy, attr_name):
    """
    Returns a new property, mapping attr_name to obj_hierarchy.attr_name.

    For example, suppose that an object of class A has an attribute b which
    itself has an attribute c which itself has an attribute d. Then placing
      e = _new_property('b.c', 'd')
    in the class definition of A makes A.e an alias for A.b.c.d
    """
    def set(self, value):
        obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        setattr(obj, attr_name, value)
    def get(self):
        obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        return getattr(obj, attr_name)
    return property(fset=set, fget=get)


def guess_units(variable):
    # works with NEURON 7.3, not with 7.1, 7.2 not tested
    nrn_units = h.units(variable.split('.')[-1])
    pq_units = nrn_units.replace("2", "**2").replace("3", "**3")
    return pq_units


class NativeCellType(BaseCellType):

    def can_record(self, variable):
        # crude check, could be improved
        return bool(recordable_pattern.match(variable))


class BaseSingleCompartmentNeuron(nrn.Section):
    """docstring"""

    def __init__(self, c_m, i_offset):

        # initialise Section object with 'pas' mechanism
        nrn.Section.__init__(self)
        self.seg = self(0.5)
        self.L = 100
        self.seg.diam = 1000/pi  # gives area = 1e-3 cm2

        self.source_section = self

        # insert current source
        self.stim = h.IClamp(0.5, sec=self)
        self.stim.delay = 0
        self.stim.dur = 1e12
        self.stim.amp = i_offset

        # for recording
        self.spike_times = h.Vector(0)
        self.traces = {}
        self.recording_time = 0

        self.v_init = None

    def area(self):
        """Membrane area in µm²"""
        return pi*self.L*self.seg.diam

    c_m      = _new_property('seg', 'cm')
    i_offset = _new_property('stim', 'amp')

    def memb_init(self):
        assert self.v_init is not None, "cell is a %s" % self.__class__.__name__
        for seg in self:
            seg.v = self.v_init
        #self.seg.v = self.v_init

    def set_parameters(self, param_dict):
        for name in self.parameter_names:
            setattr(self, name, param_dict[name])


class SingleCompartmentNeuron(BaseSingleCompartmentNeuron):
    """Single compartment with excitatory and inhibitory synapses"""

    synapse_models = {
        'current':     { 'exp': h.ExpISyn, 'alpha': h.AlphaISyn },
        'conductance': { 'exp': h.ExpSyn,  'alpha': h.AlphaSyn },
    }

    def __init__(self, syn_type, syn_shape, c_m, i_offset,
                 tau_e, tau_i, e_e, e_i):
        BaseSingleCompartmentNeuron.__init__(self, c_m, i_offset)

        self.syn_type = syn_type
        self.syn_shape = syn_shape

        # insert synapses
        assert syn_type in ('current', 'conductance'), "syn_type must be either 'current' or 'conductance'. Actual value is %s" % syn_type
        assert syn_shape in ('alpha', 'exp'), "syn_type must be either 'alpha' or 'exp'"
        synapse_model = self.synapse_models[syn_type][syn_shape]
        self.esyn = synapse_model(0.5, sec=self)
        self.isyn = synapse_model(0.5, sec=self)

    @property
    def excitatory(self):
        return self.esyn

    @property
    def inhibitory(self):
        return self.isyn

    def _get_tau_e(self):
        return self.esyn.tau
    def _set_tau_e(self, value):
        self.esyn.tau = value
    tau_e = property(fget=_get_tau_e, fset=_set_tau_e)

    def _get_tau_i(self):
        return self.isyn.tau
    def _set_tau_i(self, value):
        self.isyn.tau = value
    tau_i = property(fget=_get_tau_i, fset=_set_tau_i)

    def _get_e_e(self):
        return self.esyn.e
    def _set_e_e(self, value):
        self.esyn.e = value
    e_e = property(fget=_get_e_e, fset=_set_e_e)

    def _get_e_i(self):
        return self.isyn.e
    def _set_e_i(self, value):
        self.isyn.e = value
    e_i = property(fget=_get_e_i, fset=_set_e_i)


class LeakySingleCompartmentNeuron(SingleCompartmentNeuron):

    def __init__(self, syn_type, syn_shape, tau_m, c_m, v_rest, i_offset,
                 tau_e, tau_i, e_e, e_i):
        SingleCompartmentNeuron.__init__(self, syn_type, syn_shape, c_m, i_offset,
                                         tau_e, tau_i, e_e, e_i)
        self.insert('pas')
        self.v_init = v_rest # default value

    def __set_tau_m(self, value):
        #print("setting tau_m to", value, "cm =", self.seg.cm))
        self.seg.pas.g = 1e-3*self.seg.cm/value # cm(nF)/tau_m(ms) = G(uS) = 1e-6G(S). Divide by area (1e-3) to get factor of 1e-3
    def __get_tau_m(self):
        #print("tau_m = ", 1e-3*self.seg.cm/self.seg.pas.g, "cm = ", self.seg.cm)
        return 1e-3*self.seg.cm/self.seg.pas.g

    def __get_cm(self):
        #print("cm = ", self.seg.cm)
        return self.seg.cm
    def __set_cm(self, value): # when we set cm, need to change g to maintain the same value of tau_m
        #print("setting cm to", value)
        tau_m = self.tau_m
        self.seg.cm = value
        self.tau_m = tau_m

    v_rest = _new_property('seg.pas', 'e')
    tau_m  = property(fget=__get_tau_m, fset=__set_tau_m)
    c_m    = property(fget=__get_cm, fset=__set_cm) # if the property were called 'cm'
                                                    # it would never get accessed as the
                                                    # built-in Section.cm would always
                                                    # be used first


class StandardIF(LeakySingleCompartmentNeuron):
    """docstring"""

    def __init__(self, syn_type, syn_shape, tau_m=20, c_m=1.0, v_rest=-65,
                 v_thresh=-55, t_refrac=2, i_offset=0, v_reset=None,
                 tau_e=5, tau_i=5, e_e=0, e_i=-70):
        if v_reset is None:
            v_reset = v_rest
        LeakySingleCompartmentNeuron.__init__(self, syn_type, syn_shape, tau_m, c_m, v_rest,
                                              i_offset, tau_e, tau_i, e_e, e_i)
        # insert spike reset mechanism
        self.spike_reset = h.ResetRefrac(0.5, sec=self)
        self.spike_reset.vspike = 40 # (mV) spike height
        self.source = self.spike_reset
        self.rec = h.NetCon(self.source, None)

        # process arguments
        self.parameter_names = ['c_m', 'tau_m', 'v_rest', 'v_thresh', 't_refrac',   # 'c_m' must come before 'tau_m'
                                'i_offset', 'v_reset', 'tau_e', 'tau_i']
        if syn_type == 'conductance':
            self.parameter_names.extend(['e_e', 'e_i'])
        self.set_parameters(locals())


    v_thresh = _new_property('spike_reset', 'vthresh')
    v_reset  = _new_property('spike_reset', 'vreset')
    t_refrac = _new_property('spike_reset', 'trefrac')


class BretteGerstnerIF(LeakySingleCompartmentNeuron):
    """docstring"""

    def __init__(self, syn_type, syn_shape, tau_m=20, c_m=1.0, v_rest=-65,
                 v_thresh=-55, t_refrac=2, i_offset=0,
                 tau_e=5, tau_i=5, e_e=0, e_i=-70,
                 v_spike=0.0, v_reset=-70.6, A=4.0, B=0.0805, tau_w=144.0,
                 delta=2.0):
        LeakySingleCompartmentNeuron.__init__(self, syn_type, syn_shape, tau_m,
                                              c_m, v_rest, i_offset,
                                              tau_e, tau_i, e_e, e_i)

        # insert Brette-Gerstner spike mechanism
        self.adexp = h.AdExpIF(0.5, sec=self)
        self.source = self.adexp
        self.rec = h.NetCon(self.source, None)

        self.parameter_names = ['c_m', 'tau_m', 'v_rest', 'v_thresh', 't_refrac',
                                'i_offset', 'v_reset', 'tau_e', 'tau_i',
                                'A', 'B', 'tau_w', 'delta', 'v_spike']
        if syn_type == 'conductance':
            self.parameter_names.extend(['e_e', 'e_i'])
        self.set_parameters(locals())
        self.w_init = None

    v_thresh = _new_property('adexp', 'vthresh')
    v_reset  = _new_property('adexp', 'vreset')
    t_refrac = _new_property('adexp', 'trefrac')
    B        = _new_property('adexp',  'b')
    A        = _new_property('adexp',  'a')
    ## using 'A' because for some reason, cell.a gives the error "NameError: a, the mechanism does not exist at PySec_170bb70(0.5)"
    tau_w    = _new_property('adexp',  'tauw')
    delta    = _new_property('adexp',  'delta')

    def __set_v_spike(self, value):
        self.adexp.vspike = value
        self.adexp.vpeak = value + 10.0
    def __get_v_spike(self):
        return self.adexp.vspike
    v_spike = property(fget=__get_v_spike, fset=__set_v_spike)

    def __set_tau_m(self, value):
        self.seg.pas.g = 1e-3*self.seg.cm/value # cm(nF)/tau_m(ms) = G(uS) = 1e-6G(S). Divide by area (1e-3) to get factor of 1e-3
        self.adexp.GL = self.seg.pas.g * self.area() * 1e-2 # S/cm2 to uS
    def __get_tau_m(self):
        return 1e-3*self.seg.cm/self.seg.pas.g

    def __set_v_rest(self, value):
        self.seg.pas.e = value
        self.adexp.EL = value
    def __get_v_rest(self):
        return self.seg.pas.e
    tau_m  = property(fget=__get_tau_m, fset=__set_tau_m)
    v_rest = property(fget=__get_v_rest, fset=__set_v_rest)

    def record(self, active):
        if active:
            self.rec = h.NetCon(self.seg._ref_v, None,
                                self.get_threshold(), 0.0, 0.0,
                                sec=self)
            self.rec.record(self.spike_times)

    def get_threshold(self):
        if self.delta == 0:
            return self.adexp.vthresh
        else:
            return self.adexp.vspike

    def memb_init(self):
        assert self.v_init is not None, "cell is a %s" % self.__class__.__name__
        assert self.w_init is not None
        for seg in self:
            seg.v = self.v_init
        self.adexp.w = self.w_init


class Izhikevich_(BaseSingleCompartmentNeuron):
    """docstring"""

    def __init__(self, a_=0.02, b=0.2, c=-65.0, d=2.0, i_offset=0.0):
        BaseSingleCompartmentNeuron.__init__(self, 1.0, i_offset)
        self.L = 10
        self.seg.diam = 10/pi
        self.c_m = 1.0

        # insert Izhikevich mechanism
        self.izh = h.Izhikevich(0.5, sec=self)
        self.source = self.izh
        self.rec = h.NetCon(self.seg._ref_v, None,
                            self.get_threshold(), 0.0, 0.0,
                            sec=self)
        self.excitatory = self.inhibitory = self.source

        self.parameter_names = ['a_', 'b', 'c', 'd', 'i_offset']
        self.set_parameters(locals())
        self.u_init = None

    a_ = _new_property('izh', 'a')
    b = _new_property('izh', 'b')
    c = _new_property('izh', 'c')
    d = _new_property('izh', 'd')
    ## using 'a_' because for some reason, cell.a gives the error "NameError: a, the mechanism does not exist at PySec_170bb70(0.5)"

    def record(self, active):
        if active:
            self.rec = h.NetCon(self.seg._ref_v, None,
                                self.get_threshold(), 0.0, 0.0,
                                sec=self)
            self.rec.record(self.spike_times)

    def get_threshold(self):
        return self.izh.vthresh

    def memb_init(self):
        assert self.v_init is not None, "cell is a %s" % self.__class__.__name__
        assert self.u_init is not None
        for seg in self:
            seg.v = self.v_init
        self.izh.u = self.u_init


class GsfaGrrIF(StandardIF):
    """docstring"""

    def __init__(self, syn_type, syn_shape, tau_m=10.0, c_m=1.0, v_rest=-70.0,
                 v_thresh=-57.0, t_refrac=0.1, i_offset=0.0,
                 tau_e=1.5, tau_i=10.0, e_e=0.0, e_i=-75.0,
                 v_spike=0.0, v_reset=-70.0, q_rr=3214.0, q_sfa=14.48,
                 e_rr=-70.0, e_sfa=-70.0,
                 tau_rr=1.97, tau_sfa=110.0):

        StandardIF.__init__(self, syn_type, syn_shape, tau_m, c_m, v_rest,
                            v_thresh, t_refrac, i_offset, v_reset,
                            tau_e, tau_i, e_e, e_i)

        # insert GsfaGrr mechanism
        self.gsfa_grr = h.GsfaGrr(0.5, sec=self)
        self.v_thresh = v_thresh

        self.parameter_names = ['c_m', 'tau_m', 'v_rest', 'v_thresh', 'v_reset',
                                't_refrac', 'tau_e', 'tau_i', 'i_offset',
                                'e_rr', 'e_sfa', 'q_rr', 'q_sfa', 'tau_rr', 'tau_sfa']
        if syn_type == 'conductance':
            self.parameter_names.extend(['e_e', 'e_i'])
        self.set_parameters(locals())

    q_sfa        = _new_property('gsfa_grr',  'q_s')
    q_rr        = _new_property('gsfa_grr',  'q_r')
    tau_sfa        = _new_property('gsfa_grr',  'tau_s')
    tau_rr        = _new_property('gsfa_grr',  'tau_r')
    e_sfa = _new_property('gsfa_grr',  'E_s')
    e_rr = _new_property('gsfa_grr',  'E_r')

    def __set_v_thresh(self, value):
        self.spike_reset.vthresh = value
        # this can fail on constructor
        try:
            self.gsfa_grr.vthresh = value
        except AttributeError:
            pass

    def __get_v_thresh(self):
        return self.spike_reset.vthresh
    v_thresh = property(fget=__get_v_thresh, fset=__set_v_thresh)


class SingleCompartmentTraub(SingleCompartmentNeuron):

    def __init__(self, syn_type, syn_shape, c_m=1.0, e_leak=-65,
                 i_offset=0, tau_e=5, tau_i=5, e_e=0, e_i=-70,
                 gbar_Na=20e-3, gbar_K=6e-3, g_leak=0.01e-3, ena=50,
                 ek=-90, v_offset=-63):
        """
        Conductances are in millisiemens (S/cm2, since A = 1e-3)
        """
        SingleCompartmentNeuron.__init__(self, syn_type, syn_shape, c_m, i_offset,
                                         tau_e, tau_i, e_e, e_i)
        self.source = self.seg._ref_v
        self.source_section = self
        self.rec = h.NetCon(self.source, None, sec=self)
        self.insert('k_ion')
        self.insert('na_ion')
        self.insert('hh_traub')
        self.parameter_names = ['c_m', 'e_leak', 'i_offset', 'tau_e',
                                'tau_i', 'gbar_Na', 'gbar_K', 'g_leak', 'ena',
                                'ek', 'v_offset']
        if syn_type == 'conductance':
            self.parameter_names.extend(['e_e', 'e_i'])
        self.set_parameters(locals())
        self.v_init = e_leak # default value


    # not sure ena and ek are handled correctly

    e_leak   = _new_property('seg.hh_traub', 'el')
    v_offset = _new_property('seg.hh_traub', 'vT')
    gbar_Na  = _new_property('seg.hh_traub', 'gnabar')
    gbar_K   = _new_property('seg.hh_traub', 'gkbar')
    g_leak   = _new_property('seg.hh_traub', 'gl')

    def get_threshold(self):
        return 10.0


class RandomSpikeSource(hclass(h.NetStimFD)):

    parameter_names = ('start', '_interval', 'duration')

    def __init__(self, start=0, _interval=1e12, duration=0):
        self.start = start
        self.interval = _interval
        self.duration = duration
        self.noise = 1
        self.spike_times = h.Vector(0)
        self.source = self
        self.rec = h.NetCon(self, None)
        self.switch = h.NetCon(None, self)
        self.source_section = None
        self.seed(state.mpi_rank+state.native_rng_baseseed) # should allow user to set specific seeds somewhere, e.g. in setup()

    def _set_interval(self, value):
        self.switch.weight[0] = -1
        self.switch.event(h.t+1e-12, 0)
        self.interval = value
        self.switch.weight[0] = 1
        self.switch.event(h.t+2e-12, 1)
    def _get_interval(self):
        return self.interval
    _interval = property(fget=_get_interval, fset=_set_interval)


class VectorSpikeSource(hclass(h.VecStim)):

    parameter_names = ('spike_times',)

    def __init__(self, spike_times=[]):
        self.spike_times = spike_times
        self.source = self
        self.source_section = None
        self.rec = None

    def _set_spike_times(self, spike_times):
        # spike_times should be a Sequence object
        try:
            self._spike_times = h.Vector(spike_times.value)
        except (RuntimeError, AttributeError):
            raise errors.InvalidParameterValueError("spike_times must be an array of floats")
        self.play(self._spike_times)

    def _get_spike_times(self):
        return self._spike_times

    spike_times = property(fget=_get_spike_times,
                           fset=_set_spike_times)

    def clear_past_spikes(self):
        """If previous recordings are cleared, need to remove spikes from before the current time."""
        end = self._spike_times.indwhere(">", h.t)
        if end > 0:
            self._spike_times.remove(0, end-1)  # range is inclusive

    
