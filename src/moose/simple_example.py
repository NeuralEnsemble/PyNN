import moose
import pylab
pylab.rcParams['interactive'] = True

ms = 1e-3
mV = 1e-3

SIMDT = 0.01*ms
PLOTDT = 0.01*ms
RUNTIME = 5*ms
EREST = -70.0*mV
VLEAK = EREST + 10.613*mV
RM = 424.4e3
#RA = 7639.44e3
CM = 0.007854e-6
INJECT = 0.1e-6
VK = -90*mV
VNa = 40*mV
GK = 0.282743e-3
GNa = 0.94248e-3


model = moose.Neutral("/model")
#data = moose.Neutral("/data")
comp = moose.Compartment("/model/compartment")
comp.Rm = RM
#comp.Ra = RA
comp.Cm = CM
comp.Em = VLEAK
comp.initVm = EREST
comp.inject = INJECT

na = moose.HHChannel("na", comp)
na.Ek = VNa
na.Gbar = GNa
na.Xpower = 3
na.Ypower = 1
# (A + Bv)/(C + exp(D/E))
na.setupAlpha("X", 3.2e5 * -42*mV, -3.2e5, -1, 42*mV, -4*mV, # alpha
                  -2.8e5 * -15*mV, 2.8e5,  -1, 15*mV, 5*mV)  # beta
na.setupAlpha("Y", 128,            0,       0, 38*mV, 18*mV, # alpha
                   4.0e3,          0,       1, 15*mV, -5*mV) # beta

k = moose.HHChannel("k", comp)
k.Ek = VK
k.Gbar = GK
k.Xpower = 4
k.setupAlpha("X", 16e3 * -25*mV, -16e3, -1, 25*mV, -5*mV,
                  250          , 0,      0, 40*mV,  4*mV)

comp.connect("channel", na, "channel")
comp.connect("channel", k, "channel")

#vmTable = moose.Table("/data/Vm")
vmTable = moose.Table("/model/Vm")
vmTable.stepMode = 3
vmTable.connect("inputRequest", comp, "Vm")

ctx = moose.PyMooseBase.getContext()

#comp.getContext().setClock(0, SIMDT, 0)
#comp.getContext().setClock(1, SIMDT, 1)
#comp.getContext().setClock(2, PLOTDT, 0)
ctx.setClock(0, SIMDT, 0)
ctx.setClock(1, SIMDT, 1)
ctx.setClock(2, PLOTDT, 0)
comp.useClock(0)
comp.useClock(1, "init")
vmTable.useClock(2)


ctx.reset()
ctx.step(RUNTIME)
#comp.getContext().reset()
#comp.getContext().step(RUNTIME)
    
#vmTable.dumpFile("simple_example.v")
v = pylab.array(vmTable)/mV
t = PLOTDT*pylab.arange(0.0, v.size)/ms
pylab.plot(t, v)
pylab.xlabel("time (ms)")
pylab.ylabel("Vm (mV)")