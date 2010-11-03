"""Check that changing the spike_times of a SpikeSourceArray mid-simulation works (see ticket:166)"""

import sys
import numpy

dt = 0.1 # ms
t_step = 100.0 # ms
lag = 3.0 # ms
interactive = False

if interactive:
    import pylab
    pylab.rcParams['interactive'] = interactive

simulator_name = sys.argv[-1]
sim = __import__("pyNN.%s" % simulator_name, None, None, [simulator_name])

sim.setup(timestep=dt)

spiketimes = numpy.arange(2.0,t_step,t_step/13.0)

spikesources = sim.Population(2, sim.SpikeSourceArray)
cells = sim.Population(2, sim.IF_curr_exp)

conn = sim.Projection(spikesources, cells, sim.OneToOneConnector(weights=0.1))

cells.record_v()

spikesources[0].spike_times = spiketimes
spikesources[1].spike_times = spiketimes + lag

t = sim.run(t_step)
t = sim.run(t_step)

spiketimes += 2*t_step
spikesources[0].spike_times = spiketimes
# note we add no new spikes to the second source
t = sim.run(t_step)

final_v_0 = cells[0:1].get_v()[-1,2]
final_v_1 = cells[1:2].get_v()[-1,2]

sim.end()

if interactive:
    id, t, vtrace = cells[0:1].get_v().T
    print vtrace.shape
    print t.shape
    pylab.plot(t, vtrace)
    id, t, vtrace = cells[1:2].get_v().T
    pylab.plot(t, vtrace)

assert final_v_0 > -64.0
assert final_v_1 < -64.99