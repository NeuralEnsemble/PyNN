"""
A single-compartment Hodgkin-Huxley neuron with exponential, conductance-based
synapses, fed by a current injection.

Run as:

$ python HH_cond_exp2.py <simulator>

where <simulator> is 'neuron', 'nest', etc

Andrew Davison, UNIC, CNRS
March 2010

"""

from pyNN.utility import get_script_args

make_plot = True

simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)


setup(timestep=0.001, min_delay=0.1)

cellparams = {
    'gbar_Na': 20.0,
    'gbar_K': 6.0,
    'g_leak': 0.01,
    'cm': 0.2,
    'v_offset': -63.0,
    'e_rev_Na': 50.0,
    'e_rev_K': -90.0,
    'e_rev_leak': -65.0,
    'e_rev_E': 0.0,
    'e_rev_I': -80.0,
    'tau_syn_E': 0.2,
    'tau_syn_I': 2.0,
    'i_offset': 1.0,
}

hhcell = create(HH_cond_exp(**cellparams))
initialize(hhcell, v=-64.0)
record(['v', 'm', 'h', 'n'], hhcell, "Results/HH_cond_exp2_%s.pkl" % simulator_name)

run(20.0)

if make_plot:
    import matplotlib.pyplot as plt
    #pylab.rcParams['interactive'] = True
    plt.ion()

    data = hhcell.get_data()
    signal_names = [s.name for s in data.segments[0].analogsignals]
    vm = data.segments[0].analogsignals[signal_names.index('v')]
    plt.plot(vm.times, vm)
    plt.xlabel("time (ms)")
    plt.ylabel("Vm (mV)")

    plt.figure(2)
    for var_name in ('m', 'h', 'n'):
        signal = data.segments[0].analogsignals[signal_names.index(var_name)]
        plt.plot(signal.times, signal, label=var_name)
    plt.xlabel("time (ms)")
    plt.legend()

end()
