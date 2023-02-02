
import numpy as np
try:
    import scipy
    have_scipy = True
except ImportError:
    have_scipy = False
from numpy.testing import assert_array_equal
import quantities as pq
from pyNN.parameters import Sequence
from pyNN.errors import InvalidParameterValueError
from .fixtures import run_with_simulators
import pytest


@run_with_simulators("nest", "neuron", "brian2")
def test_EIF_cond_alpha_isfa_ista(sim, plot_figure=False):
    sim.setup(timestep=0.01, min_delay=0.1, max_delay=4.0)
    ifcell = sim.create(sim.EIF_cond_alpha_isfa_ista(
        i_offset=1.0, tau_refrac=2.0, v_spike=-40))
    ifcell.record(['spikes', 'v', 'w'])
    ifcell.initialize(v=-65, w=0)
    sim.run(200.0)
    data = ifcell.get_data().segments[0]
    expected_spike_times = np.array(
        [10.015, 25.515, 43.168, 63.41, 86.649, 113.112, 142.663, 174.76])
    if plot_figure:
        import matplotlib.pyplot as plt
        vm = data.filter(name="v")[0]
        plt.plot(vm.times, vm)
        plt.plot(expected_spike_times, -40 * np.ones_like(expected_spike_times), "ro")
        plt.savefig("test_EIF_cond_alpha_isfa_ista_%s.png" % sim.__name__)
    diff = (data.spiketrains[0].rescale(pq.ms).magnitude -
            expected_spike_times) / expected_spike_times
    assert abs(diff).max() < 0.01, abs(diff).max()
    sim.end()
    return data


@run_with_simulators("nest", "neuron", "brian2")
def test_HH_cond_exp(sim, plot_figure=False):
    sim.setup(timestep=0.001, min_delay=0.1)
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
    hhcell = sim.create(sim.HH_cond_exp(**cellparams))
    sim.initialize(hhcell, v=-64.0)
    hhcell.record('v')
    sim.run(20.0)
    v = hhcell.get_data().segments[0].filter(name='v')[0]
    if plot_figure:
        import matplotlib.pyplot as plt
        plt.plot(v.times, v)
        plt.savefig("test_HH_cond_exp_%s.png" % sim.__name__)
    sim.end()
    first_spike = v.times[np.where(v > 0)[0][0]]
    assert first_spike / pq.ms - 2.95 < 0.01


# exclude brian2 - see issue 370
@run_with_simulators("nest", "neuron")
def issue367(sim, plot_figure=False):
    # AdEx dynamics for delta_T=0
    sim.setup(timestep=0.001, min_delay=0.1, max_delay=4.0)
    v_thresh = -50
    ifcell = sim.create(sim.EIF_cond_exp_isfa_ista(
                        delta_T=0.0, i_offset=1.0, v_thresh=v_thresh, v_spike=-45))
    ifcell.record(('spikes', 'v'))
    ifcell.initialize(v=-70.6)
    sim.run(100.0)
    data = ifcell.get_data().segments[0]

    # we take the average membrane potential 0.1 ms before the spike and
    # compare it to the spike threshold
    spike_times = data.spiketrains[0]
    vm = data.analogsignals[0]
    spike_bins = ((spike_times - 0.1 * pq.ms) / vm.sampling_period).magnitude.astype(int)
    vm_before_spike = vm.magnitude[spike_bins]
    if plot_figure:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(vm.times, vm)
        plt.savefig("issue367_%s.png" % sim.__name__)
    print(sim.__name__, vm_before_spike)
    err_msg = "v_thresh = {0}, vm_before_spike.mean() = {1}".format(v_thresh,
                                                                   vm_before_spike.mean())
    assert abs((vm_before_spike.mean() - v_thresh) / v_thresh) < 0.01, err_msg
    sim.end()
    return data


@run_with_simulators("nest", "neuron", "brian2")
def test_SpikeSourcePoisson(sim, plot_figure=False):
    try:
        from scipy.stats import kstest
    except ImportError:
        pytest.skip("scipy not available")
    sim.setup()
    params = {
        "rate": [100, 200, 1000.0],
    }
    t_stop = 100000.0
    sources = sim.Population(3, sim.SpikeSourcePoisson(**params))
    sources.record('spikes')
    sim.run(t_stop)
    data = sources.get_data().segments[0]
    sim.end()

    if plot_figure:
        import matplotlib.pyplot as plt
        plt.clf()
        for i, (st, rate) in enumerate(zip(data.spiketrains, params["rate"])):
            plt.subplot(3, 1, i + 1)
            isi = st[1:] - st[:-1]
            k = rate/1000.0
            n_bins = int(np.sqrt(k * t_stop))
            values, bins, patches = plt.hist(isi, bins=n_bins,
                                             label="{} Hz".format(rate),
                                             histtype='step')
            expected = t_stop * k * (np.exp(-k * bins[:-1]) - np.exp(-k * bins[1:]))
            plt.plot((bins[1:] + bins[:-1])/2.0, expected, 'r-')
            plt.xlabel("Inter-spike interval (ms)")
            plt.legend()
        plt.savefig("test_SpikeSourcePoisson_%s.png" % sim.__name__)

    # Kolmogorov-Smirnov test
    for st, expected_rate in zip(data.spiketrains,
                                 params['rate']):
        expected_mean_isi = 1000.0/expected_rate  # ms
        isi = st[1:] - st[:-1]
        D, p = kstest(isi.magnitude,
                      "expon",
                      args=(0, expected_mean_isi),  # args are (loc, scale)
                      alternative='two-sided')
        print(expected_rate, expected_mean_isi, isi.mean(), p, D)
        assert D < 0.1

    return data


@run_with_simulators("nest", "neuron")
def test_SpikeSourceGamma(sim, plot_figure=False):
    try:
        from scipy.stats import kstest
    except ImportError:
        pytest.skip("scipy not available")
    sim.setup()
    params = {
        "beta": [100.0, 200.0, 1000.0],
        "alpha": [6, 4, 2]
    }
    t_stop = 10000.0
    sources = sim.Population(3, sim.SpikeSourceGamma(**params))
    sources.record('spikes')
    sim.run(t_stop)
    data = sources.get_data().segments[0]
    sim.end()

    if plot_figure and have_scipy:
        import matplotlib.pyplot as plt
        plt.clf()
        for i, (st, alpha, beta) in enumerate(zip(data.spiketrains, params["alpha"], params["beta"])):
            plt.subplot(3, 1, i + 1)
            isi = st[1:] - st[:-1]
            n_bins = int(np.sqrt(beta * t_stop/1000.0))
            values, bins, patches = plt.hist(isi, bins=n_bins,
                                             label="alpha={}, beta={} Hz".format(alpha, beta),
                                             histtype='step',
                                             normed=False)
            print("isi count: ", isi.size, t_stop/1000.0 * beta/alpha)
            bin_width = bins[1] - bins[0]
            expected = (t_stop * beta * bin_width) / (1000.0 * alpha) * \
                scipy.stats.gamma.pdf(bins, a=alpha, scale=1000.0/beta)
            plt.plot(bins, expected, 'r-')
            plt.xlabel("Inter-spike interval (ms)")
            plt.legend()
        plt.savefig("test_SpikeSourceGamma_%s.png" % sim.__name__)

    # Kolmogorov-Smirnov test
    print("alpha beta expected-isi actual-isi, p, D")
    for st, alpha, beta in zip(data.spiketrains,
                               params['alpha'],
                               params['beta']):
        expected_mean_isi = 1000*alpha/beta  # ms
        isi = st[1:] - st[:-1]
        # Kolmogorov-Smirnov test
        D, p = kstest(isi.magnitude,
                      "gamma",
                      args=(alpha, 0, 1000.0/beta),  # args are (a, loc, scale)
                      alternative='two-sided')
        print(alpha, beta, expected_mean_isi, isi.mean(), p, D)
        assert D < 0.1

    return data


@run_with_simulators("nest", "neuron")
def test_SpikeSourcePoissonRefractory(sim, plot_figure=False):
    try:
        from scipy.stats import kstest
    except ImportError:
        pytest.skip("scipy not available")
    sim.setup()
    params = {
        "rate": [100, 100, 50.0],
        "tau_refrac": [0.0, 5.0, 5.0]
    }
    t_stop = 100000.0
    sources = sim.Population(3, sim.SpikeSourcePoissonRefractory(**params))
    sources.record('spikes')
    sim.run(t_stop)
    data = sources.get_data().segments[0]
    sim.end()

    if plot_figure:
        import matplotlib.pyplot as plt
        plt.clf()
        for i, (st, rate, tau_refrac) in enumerate(zip(data.spiketrains,
                                                       params["rate"],
                                                       params["tau_refrac"])):
            plt.subplot(3, 1, i + 1)
            isi = st[1:] - st[:-1]
            expected_mean_isi = 1000.0/rate
            poisson_mean_isi = expected_mean_isi - tau_refrac
            k = 1/poisson_mean_isi

            n_bins = int(np.sqrt(k * t_stop))
            values, bins, patches = plt.hist(isi, bins=n_bins,
                                             label="{} Hz".format(rate),
                                             histtype='step')
            expected = t_stop/expected_mean_isi * \
                (np.exp(-(k * (bins[:-1] - tau_refrac))) -
                 np.exp(-(k * (bins[1:] - tau_refrac))))
            plt.plot((bins[1:] + bins[:-1])/2.0, expected, 'r-')
            plt.legend()
        plt.xlabel("Inter-spike interval (ms)")
        plt.savefig("test_SpikeSourcePoissonRefractory_%s.png" % sim.__name__)

    # Kolmogorov-Smirnov test
    for st, expected_rate, tau_refrac in zip(data.spiketrains,
                                             params['rate'],
                                             params['tau_refrac']):
        poisson_mean_isi = 1000.0/expected_rate - tau_refrac  # ms
        corrected_isi = (st[1:] - st[:-1]).magnitude - tau_refrac
        D, p = kstest(corrected_isi,
                      "expon",
                      args=(0, poisson_mean_isi),  # args are (loc, scale)
                      alternative='two-sided')
        print(expected_rate, poisson_mean_isi, corrected_isi.mean(), p, D)
        assert D < 0.1

    return data


@run_with_simulators("nest", "neuron", "brian2")
def test_issue511(sim):
    """Giving SpikeSourceArray an array of non-ordered spike times should produce an InvalidParameterValueError error"""
    sim.setup()
    celltype = sim.SpikeSourceArray(spike_times=[[2.4, 4.8, 6.6, 9.4], [3.5, 6.8, 9.6, 8.3]])
    with pytest.raises(InvalidParameterValueError):
        sim.Population(2, celltype)


@run_with_simulators("nest", "neuron", "brian2")
def test_update_SpikeSourceArray(sim, plot_figure=False):
    sim.setup()
    sources = sim.Population(2, sim.SpikeSourceArray(spike_times=[]))
    sources.record('spikes')
    sim.run(10.0)
    sources.set(spike_times=[
        Sequence([12, 15, 18]),
        Sequence([17, 19])
    ])
    sim.run(10.0)
    sources.set(spike_times=[
        Sequence([22, 25]),
        Sequence([23, 27, 29])
    ])
    sim.run(10.0)
    data = sources.get_data().segments[0].spiketrains
    assert_array_equal(data[0].magnitude, np.array([12, 15, 18, 22, 25]))


@run_with_simulators("nest", "neuron", "brian2")
def test_composed_neuron_model_homogeneous_receptors(sim, plot_figure=False):
    sim.setup()
    celltype1 = sim.PointNeuron(
        sim.AdExp(tau_m=10.0, v_rest=-60.0),
        AMPA=sim.AlphaPSR(tau_syn=1.0, e_syn=0.0),
        NMDA=sim.AlphaPSR(tau_syn=20.0, e_syn=0.0),
        GABAA=sim.AlphaPSR(tau_syn=1.5, e_syn=-70.0))
    celltype2 = sim.PointNeuron(
        sim.LIF(tau_m=10.0, v_rest=-60.0),
        AMPA=sim.CurrExpPostSynapticResponse(tau_syn=1.0),
        NMDA=sim.CurrExpPostSynapticResponse(tau_syn=20.0),
        GABAA=sim.CurrExpPostSynapticResponse(tau_syn=1.5))
    neurons1 = sim.Population(1, celltype1, initial_values={'v': -60.0})
    neurons2 = sim.Population(1, celltype2, initial_values={'v': -60.0})

    neurons = neurons1 + neurons2
    neurons1.record(['v', 'AMPA.gsyn', 'NMDA.gsyn', 'GABAA.gsyn'])
    neurons2.record(['v', 'AMPA.isyn', 'NMDA.isyn', 'GABAA.isyn'])

    assert neurons.get("tau_m") == pytest.approx(10.0)
    assert neurons1.get("GABAA.e_syn") == -70.0
    assert neurons2.get("GABAA.tau_syn") == 1.5

    inputs = sim.Population(
        3,
        sim.SpikeSourceArray(
            spike_times=[Sequence([30.0]), Sequence([60.0]), Sequence([90.0])]
        )
    )
    connections = {
        "AMPA1": sim.Projection(inputs[0:1], neurons1, sim.AllToAllConnector(),
                            synapse_type=sim.StaticSynapse(weight=0.01, delay=1.5),
                            receptor_type="AMPA", label="AMPA"),
        "GABAA1": sim.Projection(inputs[1:2], neurons1, sim.AllToAllConnector(),
                                synapse_type=sim.StaticSynapse(weight=0.1, delay=1.5),
                                receptor_type="GABAA", label="GABAA"),
        "NMDA1": sim.Projection(inputs[2:3], neurons1, sim.AllToAllConnector(),
                            synapse_type=sim.StaticSynapse(weight=0.005, delay=1.5),
                            receptor_type="NMDA", label="NMDA"),
        "AMPA2": sim.Projection(inputs[0:1], neurons2, sim.AllToAllConnector(),
                            synapse_type=sim.StaticSynapse(weight=0.01, delay=1.5),
                            receptor_type="AMPA", label="AMPA"),
        "GABAA2": sim.Projection(inputs[1:2], neurons2, sim.AllToAllConnector(),
                                synapse_type=sim.StaticSynapse(weight=-0.1, delay=1.5),
                                receptor_type="GABAA", label="GABAA"),
        "NMDA2": sim.Projection(inputs[2:3], neurons2, sim.AllToAllConnector(),
                            synapse_type=sim.StaticSynapse(weight=0.005, delay=1.5),
                            receptor_type="NMDA", label="NMDA")
    }
    sim.run(200.0)
    data = neurons.get_data().segments[0]
    data.filter(name='AMPA.gsyn')
    # for now, just check this runs without errors, todo: add some asserts about the data


# todo: add test of Izhikevich model


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator(("--plot-figure",
                               {"help": "generate a figure",
                                "action": "store_true"}))
    test_EIF_cond_alpha_isfa_ista(sim, plot_figure=args.plot_figure)
    test_HH_cond_exp(sim, plot_figure=args.plot_figure)
    issue367(sim, plot_figure=args.plot_figure)
    test_SpikeSourcePoisson(sim, plot_figure=args.plot_figure)
    test_SpikeSourceGamma(sim, plot_figure=args.plot_figure)
    test_SpikeSourcePoissonRefractory(sim, plot_figure=args.plot_figure)
    test_issue511(sim)
    test_update_SpikeSourceArray(sim, plot_figure=args.plot_figure)
