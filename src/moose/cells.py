"""

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import moose
import numpy
import uuid
import os

temporary_directory = "temporary_spike_files"

mV = 1e-3
ms = 1e-3
nA = 1e-9
uS = 1e-6
nF = 1e-9


class RecorderMixin(object):
    
    def record_spikes(self):
        self.spike_table = moose.Table("spikes", self)
        self.spike_table.stepMode = 4
        self.spike_table.stepSize = 0.5
        self.spike_table.useClock(0)
        self.spike_table.connect('inputRequest', self.source, 'state')
    
    def record_v(self):
        self.vmTable = moose.Table("Vm", self)
        self.vmTable.stepMode = 3
        self.vmTable.connect("inputRequest", self.comp, "Vm")
        self.vmTable.useClock(2)
        print "vmTable is at %s" % self.vmTable.path

    def record_gsyn(self, syn_name):
        syn_map = {
            'excitatory': self.esyn,
            'inhibitory': self.isyn
        }
        if not hasattr(self, "gsyn_tables"):
            self.gsyn_tables = {}
        self.gsyn_tables[syn_name] = moose.Table(syn_name, self)
        self.gsyn_tables[syn_name].stepMode = 3
        self.gsyn_tables[syn_name].connect("inputRequest", syn_map[syn_name], "Gk")
        self.gsyn_tables[syn_name].useClock(2)


class SingleCompHH(moose.Neutral, RecorderMixin):
    
    def __init__(self, path, GbarNa=20*uS, GbarK=6*uS, GLeak=0.01*uS, Cm=0.2*nF,
                 ENa=40*mV, EK=-90*mV, VLeak=-65*mV, Voff=-63*mV, ESynE=0*mV,
                 ESynI=-70*mV, tauE=2*ms, tauI=5*ms, inject=0*nA, initVm=-65*mV):
        moose.Neutral.__init__(self, path)
        self.comp = moose.Compartment("compartment", self)
        print "compartment is at %s" % self.comp.path
        print locals()
        self.comp.initVm = initVm
        self.comp.Rm = 1/GLeak
        self.comp.Cm = Cm
        self.comp.Em = VLeak
        self.comp.inject = inject
        self.na = moose.HHChannel("na", self.comp)
        self.na.Ek = ENa
        self.na.Gbar = GbarNa
        self.na.Xpower = 3
        self.na.Ypower = 1
        self.na.setupAlpha("X", 3.2e5 * (13*mV+Voff), -3.2e5, -1, -(13*mV+Voff), -4*mV, # alpha
                               -2.8e5 * (40*mV+Voff),  2.8e5, -1, -(40*mV+Voff), 5*mV)  # beta
        self.na.setupAlpha("Y", 128,                   0,      0, -(17*mV+Voff), 18*mV, # alpha
                                4.0e3,                 0,      1, -(40*mV+Voff), -5*mV) # beta

        self.k = moose.HHChannel("k", self.comp)
        self.k.Ek = EK
        self.k.Gbar = GbarK
        self.k.Xpower = 4
        self.k.setupAlpha("X", 3.2e4 * (15*mV+Voff), -3.2e4, -1, -(15*mV+Voff), -5*mV,
                               500,                  0,       0, -(10*mV+Voff),  40*mV)

        self.esyn = moose.SynChan("excitatory", self.comp)
        self.esyn.Ek = ESynE
        self.esyn.tau1 = tauE 
        self.isyn = moose.SynChan("inhibitory", self.comp)
        self.isyn.Ek = ESynI
        self.isyn.tau1 = tauI
        for syn in self.esyn, self.isyn:
            syn.tau2 = 1e-6
            syn.Gbar = 1*uS
            self.comp.connect("channel", syn, "channel")
            syn.n_incoming_connections = 0

        self.comp.connect("channel", self.na, "channel")
        self.comp.connect("channel", self.k , "channel")
        
        self.comp.useClock(0)
        self.comp.useClock(1, "init")
        
        self.source = moose.SpikeGen("source", self.comp)
        self.source.thresh = 0.0
        self.source.abs_refract = 2.0
        self.comp.connect("VmSrc", self.source, "Vm")

    # need to create some properties, so we can update parameter values after creation




class StandardIF(moose.IntFire, RecorderMixin):
    
    def __init__(self, path, syn_shape, Cm=1.0, Em=0.0, Rm=1.0, Vr=0.0, Vt=1.0,
                 refractT=0.0, inject=0.0, tau_e=0.001, tau_i=0.001, e_e=0.0,
                 e_i=-0.07):
        moose.IntFire.__init__(self, path)
        self.Cm = Cm
        self.Em = Em
        self.Rm = Rm
        self.Vr = Vr
        self.Vt = Vt
        self.refractT = refractT
        self.inject = inject
        
        self.syn_shape = syn_shape
        self.esyn = moose.SynChan("%s/excitatory" % path)
        self.isyn = moose.SynChan("%s/inhibitory" % path)
        for syn in self.esyn, self.isyn:
            syn.tau2 = 1e-6 # instantaneous rise, for shape=='exp'
            syn.Gbar = 1*uS
            self.connect("channel", syn, "channel")
            syn.n_incoming_connections = 0
        
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.e_e = e_e
        self.e_i = e_i
        
        self.source = moose.SpikeGen("source", self)
        self.source.thresh = 0.0
        self.source.abs_refract = 2.0
        self.connect("VmSrc", self.source, "Vm")
        self.comp = self # for recorder mixin
        
    def _get_tau_e(self):
        return self.esyn.tau1
    def _set_tau_e(self, val):
        self.esyn.tau1 = val
        if self.syn_shape == 'alpha':
            self.esyn.tau2 = val
    tau_e = property(fget=_get_tau_e, fset=_set_tau_e)
    
    def _get_tau_i(self):
        return self.isyn.tau1
    def _set_tau_i(self, val):
        self.isyn.tau1 = val
        if self.syn_shape == 'alpha':
            self.isyn.tau2 = val
    tau_i = property(fget=_get_tau_i, fset=_set_tau_i)
    
    def _get_e_e(self):
        return self.esyn.Ek
    def _set_e_e(self, val):
        self.esyn.Ek = val
    e_e = property(fget=_get_e_e, fset=_set_e_e)
    
    def _get_e_i(self):
        return self.isyn.Ek
    def _set_e_i(self, val):
        self.isyn.Ek = val
    e_i = property(fget=_get_e_i, fset=_set_e_i)
 


class RandomSpikeSource(moose.RandomSpike):
    
    def __init__(self, path, rate, start=0.0, duration=numpy.inf):
        moose.RandomSpike.__init__(self, path)
        self.minAmp = 1.0
        self.maxAmp = 1.0
        self.rate = rate
        self.reset = 1 #True
        self.resetValue = 0.0
        # how to handle start and duration?
        self.useClock(0)
        self.source = self

    def record_state(self):
        # for testing, can be deleted when everything is working
        self.stateTable = moose.Table("state", self)
        self.stateTable.stepMode = 3
        self.stateTable.connect("inputRequest", self, "state")
        self.stateTable.useClock(2)


class VectorSpikeSource(moose.TimeTable):
    
    def __init__(self, path, spike_times):
        moose.TimeTable.__init__(self, path)
        self._save_spikes(spike_times)
        self.source = self
        
    def _save_spikes(self, spike_times):
        ms = 1e-3
        self._spike_times = spike_times
        filename = self.filename or "%s/%s.spikes" % (temporary_directory,
                                                      uuid.uuid4().hex)
        numpy.savetxt(filename, spike_times*ms, "%g")
        self.filename = filename
        
    def _get_spike_times(self):
        return self._spike_times
    def _set_spike_times(self, spike_times):
        self._save_spikes(spike_times)
    spike_times = property(fget=_get_spike_times, fset=_set_spike_times)
