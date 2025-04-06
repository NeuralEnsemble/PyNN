from scipy.optimize import bisect
import matplotlib.pyplot as plt
import pyNN.nest as sim
from pyNN.utility.plotting import Figure, Panel

import pyNN as testpynn

spec_dict = {
    "simulation_time": 25000.0, # (ms)
    "n_ex": 16000, # Number of excitatory neurons
    "n_in": 4000, # Number of inhibitory neurons
    "r_ex": 5.0, # (Hz) Excitatory Rate
    "r_in": 20.5, # (Hz) Inhibitory Rate
    "ex_syn": 0.045, # (nA) Excitatory Synaptic Current Amplitude
    "in_syn": -0.045, # (nA) Inhibitory Synaptic Current Amplitude
    "delay": 1.0, # (ms)
    "low": 15.0, # (Hz)
    "high": 25.0, # (Hz)
    "precision": 0.01 
}

def output_rate(spec_dict, guess):

    newsim = testpynn.nest
    newsim.setup(timestep=0.1)

    cell_params = {
        "v_rest": -70.0, # (mV)
        "v_reset": -70.0, # (mV)
        "cm": 0.250, # (nF)
        "tau_m": 10, # (ms)
        "tau_refrac": 2, # (ms)
        "tau_syn_E": 2, # (ms)
        "tau_syn_I": 2, # (ms)
        "v_thresh": -55.0, # (mV)
        "i_offset": 0.0, # (nA)
    }

    cell_type = sim.IF_curr_alpha(**cell_params)
    neuron = sim.Population(1, cell_type, label="Neuron 1")
    neuron.record(["v", "spikes"])
    
    print("Inhibitory rate estimate: %5.2f Hz" % guess)
    
    noise_rate_ex = spec_dict["n_ex"] * spec_dict["r_ex"]
    noise_rate_in = spec_dict["n_in"] * spec_dict["r_in"]
    in_rate = float(abs(spec_dict["n_in"] * guess))
    
    poisson_noise_generators = sim.Population(2, sim.SpikeSourcePoisson(rate=[noise_rate_ex, in_rate]))
    
    syn = sim.StaticSynapse(delay=1)
    
    prj = sim.Projection(poisson_noise_generators, neuron, sim.AllToAllConnector(), syn)
    prj.setWeights([spec_dict["ex_syn"], spec_dict["in_syn"]])    
    
    newsim.run(1000)
    
    data_spikes = neuron.get_data(clear=True).segments[0].spiketrains[0]
    n_spikes = len(data_spikes)
    output_rate = (n_spikes * 1000.0) / spec_dict["simulation_time"]
    
    print("  -> Neuron rate: %6.2f Hz (goal: %4.2f Hz)" % (output_rate, spec_dict["r_ex"]))
    return output_rate


in_rate = bisect(lambda guess: output_rate(spec_dict, guess) - spec_dict["r_ex"], spec_dict["low"], spec_dict["high"], xtol=spec_dict["precision"])
print("Optimal rate for the inhibitory population: %.2f Hz" % in_rate)
