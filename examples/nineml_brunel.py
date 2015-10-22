# encoding: utf-8
"""
Simulation script for the Brunel (2000) network model as described in NineML.

This script imports a Python lib9ml network description from
"brunel_network_alpha.py", exports it as XML, and then
runs a simulation using the pyNN.nineml module with the NEURON
backend.

"""
from __future__ import division
import numpy as np
from neo import AnalogSignal
from quantities import ms, dimensionless
import pyNN.neuron as sim
from pyNN.nineml.read import Network
from pyNN.utility import SimulationProgressBar
from pyNN.utility.plotting import Figure, Panel
import argparse
import ninemlcatalog

cases = ['SR', 'SR2', 'SR3', 'SIfast', 'AI', 'SIslow']

parser = argparse.ArgumentParser()
parser.add_argument('case',
                    help=("The simulation case to run, can be one of '{}'"
                          .format("', '".join(cases))))
parser.add_argument('--plot', action='store_true',
                    help=("Plot the resulting figures"))
parser.add_argument('--limits', nargs=2, default=(900, 1200),
                    metavar=('LOW', 'HIGH'),
                    help="The stop and start time of the plot")
args = parser.parse_args()

sim.setup()

if args.case not in cases:
    raise Exception("Unrecognised case '{}', allowed cases are '{}'"
                    .format("', '".join(cases)))

document = ninemlcatalog.load('/network/Brunel2000/' + args.case)
xml_file = document.url

print("Building network")
net = Network(sim, xml_file)

if args.plot:
    stim = net.populations["Ext"]
    stim[:100].record('spikes')
    exc = net.populations["Exc"]
    exc.sample(50).record("spikes")
    exc.sample(3).record(["nrn_v", "syn_a"])
    inh = net.populations["Inh"]
    inh.sample(50).record("spikes")
    inh.sample(3).record(["nrn_v", "syn_a"])
else:
    all_neurons = net.assemblies["All"]
    # all.sample(50).record("spikes")
    all_neurons.record("spikes")

print("Running simulation")
t_stop = args.limits[1]
pb = SimulationProgressBar(t_stop / 80, t_stop)

sim.run(t_stop, callbacks=[pb])

print("Handling data")
if args.plot:
    stim_data = stim.get_data().segments[0]
    exc_data = exc.get_data().segments[0]
    inh_data = inh.get_data().segments[0]
else:
    all_neurons.write_data("brunel_network_alpha_%s.h5" % args.case)

sim.end()


def instantaneous_firing_rate(segment, begin, end):
    """Computed in bins of 0.1 ms """
    bins = np.arange(begin, end, 0.1)
    hist, _ = np.histogram(segment.spiketrains[0].time_slice(begin, end), bins)
    for st in segment.spiketrains[1:]:
        h, _ = np.histogram(st.time_slice(begin, end), bins)
        hist += h
    return AnalogSignal(hist, sampling_period=0.1 * ms, units=dimensionless,
                        channel_index=0, name="Spike count")

if args.plot:
    Figure(
        Panel(stim_data.spiketrains, markersize=0.2, xlim=args.limits),
        Panel(exc_data.analogsignalarrays[0], yticks=True, xlim=args.limits),
        Panel(exc_data.analogsignalarrays[1], yticks=True, xlim=args.limits),
        Panel(exc_data.spiketrains[:100], markersize=0.5, xlim=args.limits),
        Panel(instantaneous_firing_rate(exc_data, *args.limits), yticks=True),
        Panel(inh_data.analogsignalarrays[0], yticks=True, xlim=args.limits),
        Panel(inh_data.analogsignalarrays[1], yticks=True, xlim=args.limits),
        Panel(inh_data.spiketrains[:100], markersize=0.5, xlim=args.limits),
        Panel(instantaneous_firing_rate(inh_data, *args.limits), xticks=True,
              xlabel="Time (ms)", yticks=True),
    ).save("brunel_network_alpha.png")
