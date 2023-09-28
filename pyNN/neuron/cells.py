# encoding: utf-8
"""
Definition of cell classes for the neuron module.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
from math import pi, sqrt
from collections import defaultdict
from functools import reduce
import numpy as np
from neuron import h, nrn, hclass
import numpy.random
from morphio import SectionType

from .. import errors
from ..models import BaseCellType
from ..morphology import NeuriteDistribution, IonChannelDistribution
from .recording import recordable_pattern
from .simulator import state

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
    nrn_units = h.units(variable.split('.')[-1])
    pq_units = nrn_units.replace("2", "**2").replace("3", "**3")
    return pq_units


class NativeCellType(BaseCellType):

    def can_record(self, variable, location=None):
        # crude check, could be improved
        return bool(recordable_pattern.match(variable))

    # todo: use `guess_units` to construct "units" attribute


class SingleCompartmentNeuron(nrn.Section):
    """docstring"""

    def __init__(self, c_m, i_offset):

        # initialise Section object with 'pas' mechanism
        nrn.Section.__init__(self)
        self.seg = self(0.5)
        self.L = 100
        self.seg.diam = 1000 / pi  # gives area = 1e-3 cm2

        self.source_section = self

        # insert current source
        self.stim = h.IClamp(0.5, sec=self)
        self.stim.delay = 0
        self.stim.dur = 1e12
        self.stim.amp = i_offset

        # for recording
        self.spike_times = h.Vector(0)
        self.traces = defaultdict(list)
        self.recording_time = 0

        self.initial_values = {}
        self.parameters = {'c_m': c_m, 'i_offset': i_offset}

    def area(self):
        """Membrane area in µm²"""
        return pi * self.L * self.seg.diam

    c_m = _new_property('seg', 'cm')
    i_offset = _new_property('stim', 'amp')

    def memb_init(self):
        assert "v" in self.initial_values
        assert self.initial_values["v"] is not None, "cell is a %s" % self.__class__.__name__
        for seg in self:
            seg.v = self.initial_values["v"]

    def set_parameters(self):
        for name, value in self.parameters.items():
            setattr(self, name, value)


class StandardReceptorTypesMixin(object):
    """
    A mixin class to add the standard excitatory and inhibitory
    post-synaptic receptors to a model.
    """

    synapse_models = {
        'current': {'exp': h.ExpISyn, 'alpha': h.AlphaISyn},
        'conductance': {'exp': h.ExpSyn, 'alpha': h.AlphaSyn},
    }

    def __init__(self, syn_type, syn_shape, tau_e, tau_i, e_e, e_i):

        self.syn_type = syn_type
        self.syn_shape = syn_shape

        # insert synapses
        if syn_type not in ('current', 'conductance'):
            raise ValueError(
                "syn_type must be either 'current' or 'conductance'."
                f"Actual value is {syn_type}")
        if syn_shape not in ('alpha', 'exp'):
            raise ValueError("syn_type must be either 'alpha' or 'exp'")
        synapse_model = self.synapse_models[syn_type][syn_shape]
        self.esyn = synapse_model(0.5, sec=self)
        self.isyn = synapse_model(0.5, sec=self)
        self.parameters.update(tau_e=tau_e, tau_i=tau_i)
        if syn_type == 'conductance':
            self.parameters.update(e_e=e_e, e_i=e_i)

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

    def __init__(self, tau_m, c_m, v_rest, i_offset):
        SingleCompartmentNeuron.__init__(self, c_m, i_offset)
        self.insert('pas')
        self.initial_values["v"] = v_rest  # default value
        self.parameters.update(tau_m=tau_m, v_rest=v_rest)

    def __set_tau_m(self, value):
        self.seg.pas.g = 1e-3 * self.seg.cm / value

    def __get_tau_m(self):
        return 1e-3 * self.seg.cm / self.seg.pas.g

    def __get_cm(self):
        return self.seg.cm

    def __set_cm(self, value):
        # when we set cm, need to change g to maintain the same value of tau_m
        tau_m = self.tau_m
        self.seg.cm = value
        self.tau_m = tau_m

    v_rest = _new_property('seg.pas', 'e')
    tau_m = property(fget=__get_tau_m, fset=__set_tau_m)
    c_m = property(fget=__get_cm, fset=__set_cm)
    # if the property were called 'cm' it would never get accessed as the
    # built-in Section.cm would always be used first


class StandardIF(LeakySingleCompartmentNeuron):
    """docstring"""

    def __init__(self, tau_m=20, c_m=1.0, v_rest=-65,
                 v_thresh=-55, t_refrac=2, i_offset=0, v_reset=None):
        if v_reset is None:
            v_reset = v_rest
        LeakySingleCompartmentNeuron.__init__(self, tau_m, c_m, v_rest, i_offset)
        # insert spike reset mechanism
        self.spike_reset = h.ResetRefrac(0.5, sec=self)
        self.spike_reset.vspike = 40  # (mV) spike height
        self.source = self.spike_reset
        self.rec = h.NetCon(self.source, None)

        # process arguments
        self.parameters.update(v_thresh=v_thresh, t_refrac=t_refrac, v_reset=v_reset)

    v_thresh = _new_property('spike_reset', 'vthresh')
    v_reset = _new_property('spike_reset', 'vreset')
    t_refrac = _new_property('spike_reset', 'trefrac')


class StandardIFStandardReceptors(StandardIF, StandardReceptorTypesMixin):
    """docstring"""

    def __init__(self, syn_type, syn_shape, tau_m=20, c_m=1.0, v_rest=-65,
                 v_thresh=-55, t_refrac=2, i_offset=0, v_reset=None,
                 tau_e=5, tau_i=5, e_e=0, e_i=-70):
        StandardIF.__init__(self, tau_m, c_m, v_rest, v_thresh, t_refrac, i_offset, v_reset)
        StandardReceptorTypesMixin.__init__(self, syn_type, syn_shape, tau_e, tau_i, e_e, e_i)
        self.set_parameters()


class BretteGerstnerIF(LeakySingleCompartmentNeuron):
    """docstring"""

    def __init__(self, tau_m=20, c_m=1.0, v_rest=-65, v_thresh=-55, t_refrac=2,
                 i_offset=0, v_spike=0.0, v_reset=-70.6, A=4.0, B=0.0805,
                 tau_w=144.0, delta=2.0):
        LeakySingleCompartmentNeuron.__init__(self, tau_m, c_m, v_rest, i_offset)

        # insert Brette-Gerstner spike mechanism
        self.adexp = h.AdExpIF(0.5, sec=self)
        self.source = self.adexp
        self.rec = h.NetCon(self.source, None)

        local_params = locals()
        for name in ('v_thresh', 't_refrac', 'v_reset',
                     'A', 'B', 'tau_w', 'delta', 'v_spike'):
            self.parameters[name] = local_params[name]

        self.w_init = None

    v_thresh = _new_property('adexp', 'vthresh')
    v_reset = _new_property('adexp', 'vreset')
    t_refrac = _new_property('adexp', 'trefrac')
    B = _new_property('adexp', 'b')
    A = _new_property('adexp', 'a')
    # using 'A' because for some reason, cell.a gives the error "NameError: a,
    # the mechanism does not exist at PySec_170bb70(0.5)"
    tau_w = _new_property('adexp', 'tauw')
    delta = _new_property('adexp', 'delta')

    def __set_v_spike(self, value):
        self.adexp.vspike = value
        self.adexp.vpeak = value + 10.0

    def __get_v_spike(self):
        return self.adexp.vspike
    v_spike = property(fget=__get_v_spike, fset=__set_v_spike)

    def __set_tau_m(self, value):
        # cm(nF)/tau_m(ms) = G(uS) = 1e-6G(S). Divide by area (1e-3) to get factor of 1e-3
        self.seg.pas.g = 1e-3 * self.seg.cm / value
        self.adexp.GL = self.seg.pas.g * self.area() * 1e-2  # S/cm2 to uS

    def __get_tau_m(self):
        return 1e-3 * self.seg.cm / self.seg.pas.g

    def __set_v_rest(self, value):
        self.seg.pas.e = value
        self.adexp.EL = value

    def __get_v_rest(self):
        return self.seg.pas.e
    tau_m = property(fget=__get_tau_m, fset=__set_tau_m)
    v_rest = property(fget=__get_v_rest, fset=__set_v_rest)

    def get_threshold(self):
        if self.delta == 0:
            return self.adexp.vthresh
        else:
            return self.adexp.vspike

    def memb_init(self):
        assert "v" in self.initial_values
        assert "w" in self.initial_values
        assert self.initial_values["v"] is not None, "cell is a %s" % self.__class__.__name__
        assert self.initial_values["w"] is not None
        for seg in self:
            seg.v = self.initial_values["v"]
        self.adexp.w = self.initial_values["w"]


class BretteGerstnerIFStandardReceptors(BretteGerstnerIF, StandardReceptorTypesMixin):
    """docstring"""

    def __init__(self, syn_type, syn_shape, tau_m=20, c_m=1.0, v_rest=-65,
                 v_thresh=-55, t_refrac=2, i_offset=0,
                 tau_e=5, tau_i=5, e_e=0, e_i=-70,
                 v_spike=0.0, v_reset=-70.6, A=4.0, B=0.0805, tau_w=144.0,
                 delta=2.0):
        BretteGerstnerIF.__init__(self, tau_m, c_m, v_rest, v_thresh, t_refrac,
                                  i_offset, v_spike, v_reset, A, B, tau_w, delta)
        StandardReceptorTypesMixin.__init__(self, syn_type, syn_shape, tau_e, tau_i, e_e, e_i)
        self.set_parameters()


class Izhikevich_(SingleCompartmentNeuron):
    """docstring"""

    def __init__(self, a_=0.02, b=0.2, c=-65.0, d=2.0, i_offset=0.0):
        SingleCompartmentNeuron.__init__(self, 1.0, i_offset)
        self.L = 10
        self.seg.diam = 10 / pi
        self.c_m = 1.0

        # insert Izhikevich mechanism
        self.izh = h.Izhikevich(0.5, sec=self)
        self.source = self.izh
        self.rec = h.NetCon(self.seg._ref_v, None,
                            self.get_threshold(), 0.0, 0.0,
                            sec=self)
        self.excitatory = self.inhibitory = self.source

        local_params = locals()
        for name in ('a_', 'b', 'c', 'd'):
            self.parameters[name] = local_params[name]
        self.set_parameters()
        self.u_init = None

    a_ = _new_property('izh', 'a')
    b = _new_property('izh', 'b')
    c = _new_property('izh', 'c')
    d = _new_property('izh', 'd')
    # using 'a_' because for some reason, cell.a gives the error
    # "NameError: a, the mechanism does not exist at PySec_170bb70(0.5)"

    def get_threshold(self):
        return self.izh.vthresh

    def memb_init(self):
        assert "v" in self.initial_values
        assert "u" in self.initial_values
        assert self.initial_values["v"] is not None, "cell is a %s" % self.__class__.__name__
        assert self.initial_values["u"] is not None
        for seg in self:
            seg.v = self.initial_values["v"]
        self.izh.u = self.initial_values["u"]


class GsfaGrrIF(StandardIF, StandardReceptorTypesMixin):
    """docstring"""

    def __init__(self, syn_type, syn_shape, tau_m=10.0, c_m=1.0, v_rest=-70.0,
                 v_thresh=-57.0, t_refrac=0.1, i_offset=0.0,
                 tau_e=1.5, tau_i=10.0, e_e=0.0, e_i=-75.0,
                 v_spike=0.0, v_reset=-70.0, q_rr=3214.0, q_sfa=14.48,
                 e_rr=-70.0, e_sfa=-70.0,
                 tau_rr=1.97, tau_sfa=110.0):

        StandardIF.__init__(self, tau_m, c_m, v_rest,
                            v_thresh, t_refrac, i_offset, v_reset)
        StandardReceptorTypesMixin.__init__(self, syn_type, syn_shape, tau_e, tau_i, e_e, e_i)

        # insert GsfaGrr mechanism
        self.gsfa_grr = h.GsfaGrr(0.5, sec=self)
        self.v_thresh = v_thresh

        local_params = locals()
        for name in ('e_rr', 'e_sfa', 'q_rr', 'q_sfa', 'tau_rr', 'tau_sfa'):
            self.parameters[name] = local_params[name]
        self.set_parameters()

    q_sfa = _new_property('gsfa_grr', 'q_s')
    q_rr = _new_property('gsfa_grr', 'q_r')
    tau_sfa = _new_property('gsfa_grr', 'tau_s')
    tau_rr = _new_property('gsfa_grr', 'tau_r')
    e_sfa = _new_property('gsfa_grr', 'E_s')
    e_rr = _new_property('gsfa_grr', 'E_r')

    def __set_v_thresh(self, value):
        self.spike_reset.vthresh = value
        # this can fail on constructor
        # todo: figure out why it is failing and fix in a way
        #       that does not require ignoring an Exception
        try:
            self.gsfa_grr.vthresh = value
        except AttributeError:
            pass

    def __get_v_thresh(self):
        return self.spike_reset.vthresh
    v_thresh = property(fget=__get_v_thresh, fset=__set_v_thresh)


class SingleCompartmentTraub(SingleCompartmentNeuron, StandardReceptorTypesMixin):

    def __init__(self, syn_type, syn_shape, c_m=1.0, e_leak=-65,
                 i_offset=0, tau_e=5, tau_i=5, e_e=0, e_i=-70,
                 gbar_Na=20e-3, gbar_K=6e-3, g_leak=0.01e-3, ena=50,
                 ek=-90, v_offset=-63):
        """
        Conductances are in millisiemens (S/cm2, since A = 1e-3)
        """
        SingleCompartmentNeuron.__init__(self, c_m, i_offset)
        StandardReceptorTypesMixin.__init__(self, syn_type, syn_shape, tau_e, tau_i, e_e, e_i)
        self.source = self.seg._ref_v
        self.source_section = self
        self.rec = h.NetCon(self.source, None, sec=self)
        self.insert('k_ion')
        self.insert('na_ion')
        self.insert('hh_traub')

        parameter_names = ['e_leak', 'tau_e',
                           'tau_i', 'gbar_Na', 'gbar_K', 'g_leak', 'ena',
                           'ek', 'v_offset']
        local_params = locals()
        for name in parameter_names:
            self.parameters[name] = local_params[name]
        self.set_parameters()

        self.initial_values["v"] = e_leak  # default value

    # not sure ena and ek are handled correctly

    e_leak = _new_property('seg.hh_traub', 'el')
    v_offset = _new_property('seg.hh_traub', 'vT')
    gbar_Na = _new_property('seg.hh_traub', 'gnabar')
    gbar_K = _new_property('seg.hh_traub', 'gkbar')
    g_leak = _new_property('seg.hh_traub', 'gl')

    def get_threshold(self):
        return 10.0


class GIFNeuron(LeakySingleCompartmentNeuron, StandardReceptorTypesMixin):
    """
    to write...

    References:
      [1] Mensi, S., Naud, R., Pozzorini, C., Avermann, M., Petersen, C. C., &
      Gerstner, W. (2012). Parameter
      extraction and classification of three cortical neuron types reveals two
      distinct adaptation mechanisms.
      Journal of Neurophysiology, 107(6), 1756-1775.
      [2] Pozzorini, C., Mensi, S., Hagens, O., Naud, R., Koch, C., & Gerstner, W.
      (2015). Automated
      High-Throughput Characterization of Single Neurons by Means of Simplified
      Spiking Models. PLoS Comput Biol, 11(6), e1004275.
    """

    def __init__(self, syn_type, syn_shape,
                 tau_m=20, c_m=1.0, v_rest=-65,
                 t_refrac=2, i_offset=0,
                 v_reset=-55.0,
                 tau_e=5, tau_i=5, e_e=0, e_i=-70,
                 vt_star=-48.0, dV=0.5, lambda0=1.0,
                 tau_eta=(10.0, 50.0, 250.0),
                 a_eta=(0.2, 0.05, 0.025),
                 tau_gamma=(5.0, 200.0, 250.0),
                 a_gamma=(15.0, 3.0, 1.0)):

        LeakySingleCompartmentNeuron.__init__(self, tau_m, c_m, v_rest, i_offset)
        StandardReceptorTypesMixin.__init__(self, syn_type, syn_shape, tau_e, tau_i, e_e, e_i)

        self.gif_fun = h.GifCurrent(0.5, sec=self)
        self.source = self.gif_fun
        self.rec = h.NetCon(self.source, None)

        parameter_names = ['t_refrac', 'v_reset', 'tau_e', 'tau_i',
                           'vt_star', 'dV', 'lambda0', 'tau_eta', 'a_eta',
                           'tau_gamma', 'a_gamma']

        local_params = locals()
        for name in parameter_names:
            self.parameters[name] = local_params[name]
        self.set_parameters()

    def __set_tau_eta(self, value):
        self.gif_fun.tau_eta1, self.gif_fun.tau_eta2, self.gif_fun.tau_eta3 = value.value

    def __get_tau_eta(self):
        return self.gif_fun.tau_eta1, self.gif_fun.tau_eta2, self.gif_fun.tau_eta3

    tau_eta = property(fset=__set_tau_eta, fget=__get_tau_eta)

    def __set_a_eta(self, value):
        self.gif_fun.a_eta1, self.gif_fun.a_eta2, self.gif_fun.a_eta3 = value.value

    def __get_a_eta(self):
        return self.gif_fun.a_eta1, self.gif_fun.a_eta2, self.gif_fun.a_eta3

    a_eta = property(fset=__set_a_eta, fget=__get_a_eta)

    def __set_tau_gamma(self, value):
        self.gif_fun.tau_gamma1, self.gif_fun.tau_gamma2, self.gif_fun.tau_gamma3 = value.value

    def __get_tau_gamma(self):
        return self.gif_fun.tau_gamma1, self.gif_fun.tau_gamma2, self.gif_fun.tau_gamma3

    tau_gamma = property(fset=__set_tau_gamma, fget=__get_tau_gamma)

    def __set_a_gamma(self, value):
        self.gif_fun.a_gamma1, self.gif_fun.a_gamma2, self.gif_fun.a_gamma3 = value.value

    def __get_a_gamma(self):
        return self.gif_fun.a_gamma1, self.gif_fun.a_gamma2, self.gif_fun.a_gamma3

    a_gamma = property(fset=__set_a_gamma, fget=__get_a_gamma)

    v_reset = _new_property('gif_fun', 'Vr')
    t_refrac = _new_property('gif_fun', 'Tref')
    vt_star = _new_property('gif_fun', 'Vt_star')
    dV = _new_property('gif_fun', 'DV')
    lambda0 = _new_property('gif_fun', 'lambda0')

    def memb_init(self):
        for state_var in ('v', 'v_t', 'i_eta'):
            assert state_var in self.initial_values
            initial_value = self.initial_values[state_var]
            assert initial_value is not None
            if state_var == 'v':
                for seg in self:
                    seg.v = initial_value
            else:
                setattr(self.gif_fun, state_var, initial_value)


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
        # should allow user to set specific seeds somewhere, e.g. in setup()
        self.seed(state.mpi_rank + state.native_rng_baseseed)

    def __new__(cls, *arg, **kwargs):
        return super().__new__(cls, *arg, **kwargs)

    def _set_interval(self, value):
        self.switch.weight[0] = -1
        self.switch.event(h.t + 1e-12, 0)
        self.interval = value
        self.switch.weight[0] = 1
        self.switch.event(h.t + 2e-12, 1)

    def _get_interval(self):
        return self.interval
    _interval = property(fget=_get_interval, fset=_set_interval)


class RandomPoissonRefractorySpikeSource(hclass(h.PoissonStimRefractory)):

    parameter_names = ('rate', 'tau_refrac', 'start', 'duration')

    def __init__(self, rate=1, tau_refrac=0.0, start=0, duration=0):
        self.rate = rate
        self.tau_refrac = tau_refrac
        self.start = start
        self.duration = duration
        self.spike_times = h.Vector(0)
        self.source = self
        self.rec = h.NetCon(self, None)
        self.source_section = None
        self.seed(state.mpi_rank + state.native_rng_baseseed)

    def __new__(cls, *arg, **kwargs):
        return super().__new__(cls, *arg, **kwargs)


class RandomGammaSpikeSource(hclass(h.GammaStim)):

    parameter_names = ('alpha', 'beta', 'start', 'duration')

    def __init__(self, alpha=1, beta=0.1, start=0, duration=0):
        self.alpha = alpha
        self.beta = beta
        self.start = start
        self.duration = duration
        self.spike_times = h.Vector(0)
        self.source = self
        self.rec = h.NetCon(self, None)
        self.switch = h.NetCon(None, self)
        self.source_section = None
        self.seed(state.mpi_rank + state.native_rng_baseseed)

    def __new__(cls, *arg, **kwargs):
        return super().__new__(cls, *arg, **kwargs)


class VectorSpikeSource(hclass(h.VecStim)):

    parameter_names = ('spike_times',)

    def __init__(self, spike_times=[]):
        self.recording = False
        self.spike_times = spike_times
        self.source = self
        self.source_section = None
        self.rec = None
        self._recorded_spikes = np.array([])

    def __new__(cls, *arg, **kwargs):
        return super().__new__(cls, *arg, **kwargs)

    def _set_spike_times(self, spike_times):
        # spike_times should be a Sequence object
        try:
            self._spike_times = h.Vector(spike_times.value)
        except (RuntimeError, AttributeError):
            raise errors.InvalidParameterValueError("spike_times must be an array of floats")
        if np.any(spike_times.value[:-1] > spike_times.value[1:]):
            raise errors.InvalidParameterValueError(
                "Spike times given to SpikeSourceArray must be in increasing order")
        self.play(self._spike_times)
        if self.recording:
            self._recorded_spikes = np.hstack((self._recorded_spikes, spike_times.value))

    def _get_spike_times(self):
        return self._spike_times.as_numpy()

    spike_times = property(fget=_get_spike_times,
                           fset=_set_spike_times)

    @property
    def recording(self):
        return self._recording

    @recording.setter
    def recording(self, value):
        self._recording = value
        if value:
            # when we turn recording on, the cell may already have had its spike times assigned
            self._recorded_spikes = np.hstack((self._recorded_spikes, self.spike_times))

    def get_recorded_spike_times(self):
        return self._recorded_spikes

    def clear_past_spikes(self):
        """If previous recordings are cleared, need to remove spikes
        from before the current time."""
        self._recorded_spikes = self._recorded_spikes[self._recorded_spikes > h.t]


class ArtificialCell(object):
    """Wraps NEURON 'ARTIFICIAL_CELL' models for PyNN"""

    def __init__(self, mechanism_name, **parameters):
        self.source = getattr(h, mechanism_name)()
        for name, value in parameters.items():
            setattr(self.source, name, value)
        dummy = nrn.Section()

        # needed for PyNN
        self.source_section = dummy
        # todo: only need a single dummy for entire network, not one per cell
        self.parameter_names = ('tau', 'refrac')
        self.traces = defaultdict(list)
        self.spike_times = h.Vector(0)
        self.rec = h.NetCon(self.source, None)
        self.recording_time = False
        self.default = self.source
        self.initial_values = {}

    def _set_tau(self, value):
        self.source.tau = value

    def _get_tau(self):
        return self.source.tau
    tau = property(fget=_get_tau, fset=_set_tau)

    def _set_refrac(self, value):
        self.source.refrac = value

    def _get_refrac(self):
        return self.source.refrac
    refrac = property(fget=_get_refrac, fset=_set_refrac)

    # ... gkbar and g_leak properties defined similarly

    def memb_init(self):
        self.source.m = self.initial_values["m"]


class IntFire1(NativeCellType):
    default_parameters = {'tau': 10.0, 'refrac': 5.0}
    default_initial_values = {'m': 0.0}
    recordable = ['m']
    units = {'m': 'dimensionless'}
    receptor_types = ['default']
    model = ArtificialCell
    extra_parameters = {"mechanism_name": "IntFire1"}


class IntFire2(NativeCellType):
    default_parameters = {'taum': 10.0, 'taus': 20.0, 'ib': 0.0}
    default_initial_values = {'m': 0.0, 'i': 0.0}
    recordable = ['m', 'i']
    units = {'m': 'dimensionless', 'i': 'dimensionless'}
    receptor_types = ['default']
    model = ArtificialCell
    extra_parameters = {"mechanism_name": "IntFire2"}


class IntFire4(NativeCellType):
    default_parameters = {
        'taum': 50.0,
        'taue': 5.0,
        'taui1': 10.0,
        'taui2': 20.0,
    }
    default_initial_values = {'m': 0.0, 'e': 0.0, 'i1': 0.0, 'i2': 0.0}
    recordable = ['e', 'i1', 'i2', 'm']
    units = {'e': 'dimensionless',
             'i1': 'dimensionless',
             'i2': 'dimensionless',
             'm': 'dimensionless'}
    receptor_types = ['default']
    model = ArtificialCell
    extra_parameters = {"mechanism_name": "IntFire4"}


PROXIMAL = 0
DISTAL = 1


class NeuronTemplate(object):

    def __init__(self, morphology, cm, Ra, ionic_species, **other_parameters):
        import neuroml
        import neuroml.arraymorph
        from morphio import Morphology as MorphIOMorphology

        self.initial_values = defaultdict(dict)
        self.traces = defaultdict(list)
        self.recording_time = False
        self.spike_source = None
        self.spike_times = h.Vector(0)

        # create morphology
        self.morphology = morphology
        self.ionic_species = ionic_species
        self.sections = {}
        self.section_labels = defaultdict(set)
        self.synaptic_receptors = {}
        for receptor_name in self.post_synaptic_entities:
            self.synaptic_receptors[receptor_name] = defaultdict(list)
        self.locations = {}  # to store recording and current injection locations

        d_lambda = 0.1

        def lambda_f(freq, section):
            return 1e5 * sqrt(section.diam / (4 * pi * freq * section.Ra * section.cm))

        if isinstance(morphology._morphology, neuroml.arraymorph.ArrayMorphology):
            M = morphology._morphology
            for i in range(len(morphology._morphology)):
                vertex = M.vertices[i]
                parent_index = M.connectivity[i]
                parent = M.vertices[parent_index]
                section = nrn.Section()
                for v in (vertex, parent):
                    x, y, z, d = v
                    h.pt3dadd(x, y, z, d, sec=section)
                section.nseg = 1 + 2 * int((0.999 + section.L/(d_lambda * lambda_f(100, section)))/2)
                section.cm = cm
                section.Ra = Ra
                # ignore fractions_along for now
                if i > 1:
                    section.connect(self.sections[parent_index], DISTAL, PROXIMAL)
                self.sections[i] = section
            self.morphology._soma_index = 0  # fragile temporary hack - should be index of the vertex with no parent
        elif isinstance(morphology._morphology, neuroml.Morphology):
            unresolved_connections = []
            for index, segment in enumerate(morphology.segments):
                section = nrn.Section(name=segment.name)
                for p in (segment.proximal, segment.distal):
                    h.pt3dadd(p.x, p.y, p.z, p.diameter, sec=section)
                if isinstance(cm, NeuriteDistribution):
                    section.cm = cm.value_in(self.morphology, index)
                else:
                    section.cm = cm
                section.Ra = Ra
                section.nseg = 1 + 2 * int((0.999 + section.L/(d_lambda * lambda_f(100, section)))/2)
                segment_id = segment.id
                assert segment_id is not None
                if segment.parent:
                    parent_id = segment.parent.id
                    connection_point = DISTAL  # should generalize
                    if segment.parent.id in self.sections:
                        section.connect(self.sections[parent_id], connection_point, PROXIMAL)
                    else:
                        unresolved_connections.append((segment_id, parent_id))
                self.sections[segment_id] = section
                if segment.name == "soma":
                    self.morphology._soma_index = segment_id
                if segment.name is not None:
                    self.section_labels[segment.name].add(segment_id)
                segment._section = section
            for section_id, parent_id in unresolved_connections:
                self.sections[section_id].connect(self.sections[parent_id], DISTAL, PROXIMAL)
        elif isinstance(morphology._morphology, MorphIOMorphology):
            m = morphology._morphology
            soma = nrn.Section(name="soma")
            self.sections[-1] = soma
            self.section_labels["soma"].add(-1)
            self.morphology._soma_index = 0
            if isinstance(cm, NeuriteDistribution):
                soma.cm = cm.value_in(self.morphology, "soma")
            else:
                soma.cm = cm
            soma.Ra = Ra
            for (x, y, z), d in zip(m.soma.points, m.soma.diameters):
                h.pt3dadd(x, y, z, d, sec=soma)
            for root_section in m.root_sections:
                for m_section in root_section.iter():
                    nrn_section = nrn.Section(name=f"section_{m_section.id}")
                    for (x, y, z), d in zip(m_section.points, m_section.diameters):
                        h.pt3dadd(x, y, z, d, sec=nrn_section)
                    nrn_section.nseg = 1 + 2 * int((0.999 + nrn_section.L/(d_lambda * lambda_f(100, nrn_section)))/2)
                    if isinstance(cm, NeuriteDistribution):
                        nrn_section.cm = cm.value_in(self.morphology, section.id)
                    else:
                        nrn_section.cm = cm
                    nrn_section.Ra = Ra
                    if m_section.is_root:
                        nrn_section.connect(soma, DISTAL, PROXIMAL)
                        # todo: connect basal dendrites, axon, apical dendrites to different points on the soma
                    else:
                        nrn_section.connect(self.sections[m_section.parent.id], DISTAL, PROXIMAL)
                    self.sections[m_section.id] = nrn_section
                    self.section_labels[m_section.type.name].add(m_section.id)
        else:
            raise ValueError("{} not supported as a neuron morphology".format(type(morphology)))

        # insert ion channels
        for name, ion_channel in self.ion_channels.items():
            parameters = other_parameters[name]
            mechanism_name = ion_channel.model
            conductance_density = parameters[ion_channel.conductance_density_parameter]
            for index, id in enumerate(self.sections):
                #if id == -1:
                if isinstance(conductance_density, float):
                    g = conductance_density
                elif isinstance(conductance_density, IonChannelDistribution):
                    g = conductance_density.value_in(self.morphology, index)
                else:
                    raise TypeError("Conductance density should be a float or an IonChannelDistribution object")
                if g is not None and g > 0:
                    section = self.sections[id]
                    section.insert(mechanism_name)
                    varname = ion_channel.conductance_density_parameter + "_" + ion_channel.model
                    setattr(section, varname, g)
                    # We're not using the leak conductance from the hh mechanism,
                    # so set the conductance to zero
                    if mechanism_name == "hh":
                        setattr(section, "gl_hh", 0.0)
                    for param_name, value in parameters.items():
                        if param_name != ion_channel.conductance_density_parameter:
                            if isinstance(value, IonChannelDistribution):
                                value = value.value_in(self.morphology, index)
                            try:
                                setattr(section, param_name + "_" + ion_channel.model, value)
                            except AttributeError:  # e.g. parameters not defined within a mechanism, e.g. ena
                                setattr(section, param_name, value)
                            ##print(index, mechanism_name, param_name, value)

        # insert post-synaptic mechanisms
        for name, pse in self.post_synaptic_entities.items():
            parameters = other_parameters[name]
            synapse_model = pse.model
            location_generator = parameters.pop("locations")
            for location_label in location_generator.generate_locations(self.morphology, label_prefix=name, cell=self):
                location = self.locations[location_label]
                section, section_id, position = location.get_section_and_position()
                syn_obj = synapse_model(position, sec=section)
                self.synaptic_receptors[name][section_id].append(syn_obj)
                for pname, pvalue in parameters.items():
                    setattr(syn_obj, pname, pvalue)

        # handle ionic species
        def set_in_section(section, index, name, value):
            if isinstance(value, IonChannelDistribution):  # should be "NeuriteDistribution"
                value = value.value_in(self.morphology, index)
            if value is not None:
                if name == "eca":     # tmp hack
                    section.push()
                    h.ion_style("ca_ion", 1, 1, 0, 1, 0)
                    h.pop_section()
                try:
                    setattr(section, name, value)
                except (NameError, AttributeError) as err:  # section does not contain ion
                    if "the mechanism does not exist" not in str(err):
                        raise

        for ion_name, parameters in self.ionic_species.items():
            for index, id in enumerate(self.sections):
                section = self.sections[id]
                set_in_section(section, index, "e{}".format(ion_name), parameters.reversal_potential)
                if parameters.internal_concentration:
                    set_in_section(section, index, "{}i".format(ion_name), parameters.internal_concentration)
                if parameters.external_concentration:
                    set_in_section(section, index, "{}o".format(ion_name), parameters.external_concentration)

        # set source section
        if self.spike_source:
            self.source_section = self.sections[self.spike_source]
        elif "axon_initial_segment" in self.sections:
            self.source_section = self.sections["axon_initial_segment"]
        else:
            self.source_section = self.sections[morphology.soma_index]
        self.source = self.source_section(0.5)._ref_v
        self.rec = h.NetCon(self.source, None, sec=self.source_section)

    def memb_init(self):
        # initialize membrane potential
        initial_value = self.initial_values["v"]
        assert initial_value is not None
        for section in self.sections.values():
            for seg in section:
                seg.v = initial_value
        # initialize state variables
        for channel_name, channel_obj in self.ion_channels.items():
            for std_state_name, (mech_name, mech_state_name) in channel_obj.variable_translations.items():
                initial_value = self.initial_values[channel_name].get(std_state_name, None)
                if initial_value is not None:
                    for section in self.sections.values():
                        for seg in section:
                            try:
                                mechanism = getattr(seg, mech_name)  # e.g. "hh"
                            except Exception:  # todo: catch specific NEURON RuntimeError
                                pass
                            else:
                                setattr(mechanism, mech_state_name, initial_value)
        # todo: synaptic state variables?