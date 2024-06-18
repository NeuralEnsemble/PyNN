import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel
import pyNN.nest as sim
import pyNN as devpynn

timesteps = [0.1, 0.5, 1.0]
fig_counter=0

for dt in timesteps:
    newsim = devpynn.nest
    newsim.setup(timestep=dt)

    cell_params = {
        "v_rest": -70.0, # (mV)
        "v_reset": -70.0, # (mV)
        "cm": 0.250, # (nF)
        "tau_m": 10, # (ms)
        "tau_refrac": 2, # (ms)
        "tau_syn_E": 2, # (ms)
        "tau_syn_I": 2, # (ms)
        "v_thresh": -55.0, # (mV)
        "i_offset": 0.376 # (nA)
    }

    cell_type = sim.IF_curr_alpha(**cell_params)
    neuron = sim.Population(1, cell_type, label="Neuron 1")
    neuron.record(["v", "spikes"])
    newsim.run(1000.0)

    data_v = neuron.get_data().segments[0].filter(name="v")[0]
    data_spikes = neuron.get_data().segments[0].spiketrains[0]

    fig_counter += 1
    plt.figure(fig_counter)
    plt.plot(data_v[:,0].times, data_v[:, 0])
    plt.xlabel("Time (in ms)")
    plt.ylabel("Membrane Potential (in mV)")
    plt.savefig(f"PLOT_iaf_neuron_current_{fig_counter}")
    plt.show()

    print(f"\nNumber of spikes: {len(data_spikes)}")