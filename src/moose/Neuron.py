import moose
class SingleCompHH(object):
	def __init__(self, path, GbarNa, GbarK, GLeak, Cm, ENa, EK, VLeak, Voff, ESynE, ESynI, tauE, tauI, inject, initVm):
		self.comp = moose.Compartment( path )
		self.comp.initVm = initVm
		self.comp.Rm = 1/GLeak
		self.Cm = Cm
		self.Em = VLeak
		self.inject = inject
		self.na = moose.HHChannel(path+"/na")
		self.na.Ek = ENa
		self.na.Gbar = GbarNa
		self.k = moose.HHChannel(path+"/k")
		self.k.Ek = EK
		self.k.Gbar = GbarK
		self.synE = moose.SynChan(path+"/excitatory")
		self.synE.Ek = ESynE
		self.synE.tau1 = 1e-6
		self.synE.tau2 = tauE
		self.synE.Gbar = 1e-9
		self.synI = moose.SynChan(path+"/inhibitory")
		self.synI.Ek = ESynI
		self.synI.tau1 = 1e-6
		self.synI.tau2 = tauI
		self.synI.Gbar = 1e-9

		self.comp.connect("channel", self.synE, "channel")
		self.comp.connect("channel", self.synI, "channel")
		self.comp.connect("channel", self.na, "channel")
		self.comp.connect("channel", self.k , "channel")

#		self.vmTable = moose.Table(path+"/vm")
#		self.vmTable.connect("inputRequest", self.comp, "Vm")
#		self.vmTable.stepmode = 3
#		moose.PyMooseBase.getContext().useClock(path+","+path+"/#")

a = SingleCompHH("/comp", 1e-9, 1e-9, 1e-9, 1e-12, -0.06, 0.08, -0.06, 0.001, 0.05, -0.05, 1e-2, 2e-2, 1e-9, -0.06)
print "Successfully setup SingleCompHH"
