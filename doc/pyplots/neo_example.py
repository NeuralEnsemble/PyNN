import pyNN.neuron as sim  # can of course replace `neuron` with `nest`, `brian`, etc.
import matplotlib.pyplot as plt
import numpy as np

sim.setup(timestep=0.01)
p_in = sim.Population(10, sim.SpikeSourcePoisson(rate=10.0), label="input")
p_out = sim.Population(10, sim.EIF_cond_exp_isfa_ista(), label="AdExp neurons")

syn = sim.StaticSynapse(weight=0.05)
random = sim.FixedProbabilityConnector(p_connect=0.5)
connections = sim.Projection(p_in, p_out, random, syn, receptor_type='excitatory')

p_in.record('spikes')
p_out.record('spikes')                    # record spikes from all neurons
p_out[0:2].record(['v', 'w', 'gsyn_exc'])  # record other variables from first two neurons

sim.run(500.0)

spikes_in = p_in.get_data()
data_out = p_out.get_data()

fig_settings = {
    'lines.linewidth': 0.5,
    'axes.linewidth': 0.5,
    'axes.labelsize': 'small',
    'legend.fontsize': 'small',
    'font.size': 8
}
plt.rcParams.update(fig_settings)
plt.figure(1, figsize=(6, 8))


def plot_spiketrains(segment):
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '.')
        plt.ylabel(segment.name)
        plt.setp(plt.gca().get_xticklabels(), visible=False)


def plot_signal(signal, index, colour='b'):
    label = "Neuron %d" % signal.annotations['source_ids'][index]
    plt.plot(signal.times, signal[:, index], colour, label=label)
    plt.ylabel("%s (%s)" % (signal.name, signal.units._dimensionality.string))
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.legend()


n_panels = sum(a.shape[1] for a in data_out.segments[0].analogsignals) + 2
plt.subplot(n_panels, 1, 1)
plot_spiketrains(spikes_in.segments[0])
plt.subplot(n_panels, 1, 2)
plot_spiketrains(data_out.segments[0])
panel = 3
for array in data_out.segments[0].analogsignals:
    for i in range(array.shape[1]):
        plt.subplot(n_panels, 1, panel)
        plot_signal(array, i, colour='bg'[panel % 2])
        panel += 1
plt.xlabel("time (%s)" % array.times.units._dimensionality.string)
plt.setp(plt.gca().get_xticklabels(), visible=True)

plt.show()
