"""Check that changing the spike_times of a SpikeSourceArray mid-simulation works."""

import sys

dt = 0.1
t_step = 100.0
interactive = False

if interactive:
    import pylab
    pylab.rcParams['interactive'] = interactive

simulator = sys.argv[-1]
sim = __import__("pyNN.%s" % simulator, None, None, [simulator])

sim.setup(dt=dt)

spiketimes = pylab.arange(2.0,t_step,t_step/13.0)

cell1 = sim.create(sim.SpikeSourceArray) #, {'spike_times': spiketimes})
cell2 = sim.create(sim.IF_curr_exp)

sim.connect(cell1, cell2, weight=0.1)
sim.record(cell1, "incremental_spiketimes_%s.ras" % simulator)
sim.record_v(cell2, "incremental_spiketimes_%s.v" % simulator)

cell1.spike_times = spiketimes

t = sim.run(t_step)
print "t = ", t
t = sim.run(t_step)

spiketimes += 2*t_step
cell1.spike_times = spiketimes
t = sim.run(t_step)

sim.end()

if interactive:
    vtrace = pylab.load("incremental_spiketimes_%s.v" % simulator)[:,0]
    t = pylab.arange(0, 3*t_step+dt, dt)[:len(vtrace)]
    print vtrace.shape
    print t.shape
    pylab.plot(t, vtrace)
