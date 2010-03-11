import moose
from pyNN import standardmodels, cells

mV = 1e-3
ms = 1e-3
nA = 1e-9

class SingleCompHH(moose.Neutral):
    
    def __init__(self, path, GbarNa=0.0, GbarK=0.0, GLeak=0.0, Cm=1.0,
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

        #self.synE = moose.SynChan("excitatory", self.comp)
        #self.synE.Ek = ESynE
        #self.synE.tau1 = 1e-6
        #self.synE.tau2 = tauE
        #self.synE.Gbar = 1e-9
        #self.synI = moose.SynChan("inhibitory", self.comp)
        #self.synI.Ek = ESynI
        #self.synI.tau1 = 1e-6
        #self.synI.tau2 = tauI
        #self.synI.Gbar = 1e-9
    
        #self.comp.connect("channel", self.synE, "channel")
        #self.comp.connect("channel", self.synI, "channel")
        self.comp.connect("channel", self.na, "channel")
        self.comp.connect("channel", self.k , "channel")
        
        self.comp.useClock(0)
        self.comp.useClock(1, "init")

    def record_v(self):
        self.vmTable = moose.Table("Vm", self)
        self.vmTable.stepMode = 3
        self.vmTable.connect("inputRequest", self.comp, "Vm")
        self.vmTable.useClock(2)
        print "vmTable is at %s" % self.vmTable.path
        #moose.PyMooseBase.getContext().useClock(0, self.comp.path+"/##")

#a = SingleCompHH("/comp", 1e-9, 1e-9, 1e-9, 1e-12, -0.06, 0.08, -0.06, 0.001, 0.05, -0.05, 1e-2, 2e-2, 1e-9, -0.06)
#print "Successfully setup SingleCompHH"



class HH_cond_exp(cells.HH_cond_exp):
    """Single compartment cell with an Na channel and a K channel"""
    translations = standardmodels.build_translations(
        ('gbar_Na',    'GbarNa', 1e-9),   
        ('gbar_K',     'GbarK', 1e-9),    
        ('g_leak',     'GLeak', 1e-9),    
        ('cm',         'Cm',    1e-9),  
        ('v_offset',   'Voff', 1e-3),
        ('e_rev_Na',   'ENa', 1e-3),
        ('e_rev_K',    'EK', 1e-3), 
        ('e_rev_leak', 'VLeak', 1e-3),
        ('e_rev_E',    'ESynE', 1e-3),
        ('e_rev_I',    'ESynI', 1e-3),
        ('tau_syn_E',  'tauE', 1e-3),
        ('tau_syn_I',  'tauI', 1e-3),
        ('i_offset',   'inject', 1e-9),
        ('v_init',     'initVm', 1e-3),
    )
    model = SingleCompHH











