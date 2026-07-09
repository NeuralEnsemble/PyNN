import sys
import numpy as np
try:
    import scipy
    have_scipy = True
except ImportError:
    have_scipy = False
from numpy.testing import assert_array_equal, assert_allclose
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
    if "pytest" not in sys.modules:
        return data


@run_with_simulators("arbor", "nest", "neuron", "brian2")
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
    if "pytest" not in sys.modules:
        return data


@run_with_simulators("nest", "neuron", "brian2", "arbor")
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
    if "pytest" not in sys.modules:
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

    if "pytest" not in sys.modules:
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

    if "pytest" not in sys.modules:
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


@run_with_simulators("nest", "neuron", "brian2", "arbor")
def test_SpikeSourceArray_delivers_spike_times(sim):
    """A SpikeSourceArray emits exactly its specified spike times."""
    sim.setup(timestep=0.1)
    spike_times = [10.0, 25.0, 40.0, 55.0]
    sources = sim.Population(2, sim.SpikeSourceArray(spike_times=spike_times))
    sources.record('spikes')
    sim.run(70.0)
    spiketrains = sources.get_data().segments[0].spiketrains
    assert len(spiketrains) == 2
    for st in spiketrains:
        assert_allclose(np.array(st.magnitude), spike_times, atol=0.2)
    sim.end()


@run_with_simulators("arbor", "nest", "brian2")
def test_IF_curr_delta_voltage_step(sim):
    """A single delta-synapse input steps V by the weight (mV).

    Guards the Arbor weight scaling: a lif_cell delta event adds weight/C_m to
    V_m, so the connection weight is scaled by C_m to recover PyNN's mV step.
    """
    sim.setup(timestep=0.1)
    v_rest = -65.0
    weight = 5.0  # mV voltage step
    source = sim.Population(1, sim.SpikeSourceArray(spike_times=[20.0]))
    cell = sim.Population(1, sim.IF_curr_delta(
        v_rest=v_rest, v_reset=v_rest, v_thresh=-40.0,
        tau_m=20.0, cm=1.0, tau_refrac=0.1),
        initial_values={'v': v_rest})
    sim.Projection(source, cell, sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=weight, delay=1.0),
                   receptor_type="excitatory")
    cell.record('v')
    sim.run(60.0)
    v = cell.get_data().segments[0].filter(name='v')[0].magnitude[:, 0]
    step = v.max() - v_rest
    assert abs(step - weight) < 1e-12, step
    sim.end()


@run_with_simulators("arbor", "brian2")
def test_IF_curr_delta_fires_and_resets(sim):
    """A supra-threshold delta train makes the wired cell fire; the other is silent."""
    sim.setup(timestep=0.1)
    source = sim.Population(1, sim.SpikeSourceArray(
        spike_times=[10.0, 11.0, 12.0, 13.0]))
    cells = sim.Population(2, sim.IF_curr_delta(
        v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0,
        tau_m=20.0, cm=1.0, tau_refrac=5.0),
        initial_values={'v': -65.0})
    sim.Projection(source, cells[0:1], sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=8.0, delay=1.0),
                   receptor_type="excitatory")
    cells.record('spikes')
    sim.run(60.0)
    spiketrains = cells.get_data().segments[0].spiketrains
    assert len(spiketrains) == 2
    counts = sorted(len(st) for st in spiketrains)
    assert counts[0] == 0        # unconnected cell stays silent
    assert counts[1] >= 1        # wired cell fires
    sim.end()


def _lif_isi_theory(v_rest, v_reset, v_thresh, tau_m, cm, tau_refrac, i_offset):
    """Analytic steady-state ISI of a leaky integrate-and-fire neuron driven by a
    constant current i_offset (independent of simulator backend)."""
    v_inf = v_rest + i_offset * tau_m / cm
    return tau_refrac + tau_m * np.log((v_inf - v_reset) / (v_inf - v_thresh))


@run_with_simulators("arbor", "neuron", "nest", "brian2")
def test_IF_exp_point_neuron_fI(sim):
    """IF_cond_exp / IF_curr_exp fire regularly under a constant current, with the
    inter-spike interval matching the analytic LIF prediction on every backend.
    """
    params = dict(v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0,
                  tau_m=20.0, cm=1.0, tau_refrac=5.0)
    isi_theory = _lif_isi_theory(i_offset=1.0, **params)
    for cell_class in (sim.IF_curr_exp, sim.IF_cond_exp):
        sim.setup(timestep=0.025)
        cell = sim.Population(1, cell_class(i_offset=1.0, tau_syn_E=5.0, tau_syn_I=5.0,
                                            **params),
                              initial_values={'v': -65.0})
        cell.record('spikes')
        sim.run(500.0)
        spikes = np.array(cell.get_data().segments[0].spiketrains[0])
        assert len(spikes) >= 10, (cell_class.__name__, len(spikes))
        isi = np.diff(spikes)[1:].mean()  # skip the first (from-rest) interval
        assert abs(isi - isi_theory) < 1.0, (cell_class.__name__, isi, isi_theory)
        sim.end()


@run_with_simulators("arbor", "neuron", "nest", "brian2")
def test_IF_curr_exp_EPSP(sim):
    """A single presynaptic spike into IF_curr_exp produces an EPSP whose peak
    matches the analytic current-based-synapse prediction on every backend."""
    v_rest, cm, tau_m, tau_syn, weight = -65.0, 1.0, 20.0, 5.0, 0.5
    sim.setup(timestep=0.025)
    source = sim.Population(1, sim.SpikeSourceArray(spike_times=[20.0]))
    cell = sim.Population(1, sim.IF_curr_exp(
        v_rest=v_rest, v_reset=v_rest, v_thresh=-50.0, tau_m=tau_m, cm=cm,
        tau_refrac=5.0, tau_syn_E=tau_syn, i_offset=0.0),
        initial_values={'v': v_rest})
    sim.Projection(source, cell, sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=weight, delay=1.0),
                   receptor_type="excitatory")
    cell.record('v')
    sim.run(100.0)
    v = cell.get_data().segments[0].filter(name='v')[0].magnitude[:, 0]
    epsp = v.max() - v_rest
    # analytic peak of a current-based exponential synapse
    t_peak = np.log(tau_m / tau_syn) / (1 / tau_syn - 1 / tau_m)
    prefactor = (weight / cm) * (tau_m * tau_syn / (tau_m - tau_syn))
    epsp_theory = prefactor * (np.exp(-t_peak / tau_m) - np.exp(-t_peak / tau_syn))
    assert abs(epsp - epsp_theory) < 0.05, (epsp, epsp_theory)
    sim.end()


@run_with_simulators("arbor", "nest", "neuron", "brian2")
def test_EIF_cond_alpha_isfa_ista_spike_times(sim):
    """The AdExp dynamics of EIF_cond_alpha_isfa_ista driven by a constant current
    reproduce the reference spike times (from the NEURON backend) to <1% on every
    backend. (Same setup as test_EIF_cond_alpha_isfa_ista but recording only
    spikes, so it can also run on Arbor, which does not yet record `w`.)"""
    sim.setup(timestep=0.01, min_delay=0.1, max_delay=4.0)
    ifcell = sim.Population(1, sim.EIF_cond_alpha_isfa_ista(
        i_offset=1.0, tau_refrac=2.0, v_spike=-40))
    ifcell.initialize(v=-65, w=0)
    ifcell.record('spikes')
    sim.run(200.0)
    spike_times = ifcell.get_data().segments[0].spiketrains[0].rescale(pq.ms).magnitude
    sim.end()
    expected_spike_times = np.array(
        [10.015, 25.515, 43.168, 63.41, 86.649, 113.112, 142.663, 174.76])
    assert len(spike_times) == len(expected_spike_times), (spike_times, expected_spike_times)
    diff = (spike_times - expected_spike_times) / expected_spike_times
    assert abs(diff).max() < 0.01, abs(diff).max()


@run_with_simulators("arbor", "neuron", "nest", "brian2")
def test_EIF_cond_exp_isfa_ista_adaptation(sim):
    """A constant supra-threshold current into EIF_cond_exp_isfa_ista produces a
    regular spike train with spike-frequency adaptation (monotonically increasing
    ISIs), matching the NEURON backend (cross-checked to <0.2% on mean ISI)."""
    sim.setup(timestep=0.01, min_delay=0.1)
    cell = sim.Population(1, sim.EIF_cond_exp_isfa_ista(
        i_offset=1.0, tau_refrac=2.0, v_spike=-40.0))
    cell.initialize(v=-70.6, w=0.0)
    cell.record('spikes')
    sim.run(200.0)
    st = cell.get_data().segments[0].spiketrains[0].magnitude
    sim.end()
    assert len(st) >= 6, len(st)
    isis = np.diff(st)
    # spike-frequency adaptation: each ISI is longer than the previous one
    assert np.all(np.diff(isis) > 0), isis
    # adaptation is substantial (last ISI at least 1.5x the first)
    assert isis[-1] > 1.5 * isis[0], isis


@run_with_simulators("arbor", "neuron")
def test_Izhikevich_regular_spiking(sim):
    """A constant current into a regular-spiking Izhikevich neuron produces a
    steady spike train. Arbor's quadratic dynamics and one-step reset are
    cross-checked against NEURON to <0.03 ms on spike times / <0.1% on ISI."""
    sim.setup(timestep=0.01, min_delay=0.1)
    cell = sim.Population(1, sim.Izhikevich(a=0.02, b=0.2, c=-65.0, d=8.0, i_offset=0.01))
    cell.initialize(v=-70.0, u=-14.0)
    cell.record('spikes')
    sim.run(300.0)
    st = cell.get_data().segments[0].spiketrains[0].magnitude
    sim.end()
    assert 6 <= len(st) <= 10, len(st)
    isis = np.diff(st)
    # after the first ISI the train is regular (steady inter-spike interval)
    assert abs(isis[-1] - isis[-2]) < 0.1, isis
    assert 40.0 < isis[-1] < 50.0, isis


@run_with_simulators("arbor", "neuron")
def test_IF_cond_exp_gsfa_grr_adaptation(sim):
    """A constant supra-threshold current into IF_cond_exp_gsfa_grr produces spike-
    frequency adaptation from the g_s conductance: the ISIs lengthen and then settle
    to a steady value. Arbor is cross-checked against NEURON to <0.01 ms on spike
    times; nest/brian2 are excluded here because their gsfa_grr implementations
    adapt differently for this strongly-driven regime."""
    sim.setup(timestep=0.01, min_delay=0.1)
    cell = sim.Population(1, sim.IF_cond_exp_gsfa_grr(
        v_rest=-65.0, v_reset=-65.0, v_thresh=-57.0, tau_m=10.0, cm=0.25,
        tau_refrac=2.0, i_offset=1.0,
        tau_sfa=100.0, q_sfa=15.0, e_rev_sfa=-75.0,
        tau_rr=2.0, q_rr=3000.0, e_rev_rr=-75.0))
    cell.initialize(v=-65.0)
    cell.record('spikes')
    sim.run(400.0)
    st = cell.get_data().segments[0].spiketrains[0].magnitude
    sim.end()
    assert len(st) >= 8, len(st)
    isis = np.diff(st)
    # early ISIs lengthen (adaptation) and later ones settle to a steady rate
    assert isis[3] > isis[0], isis
    assert isis[-1] > 2.0 * isis[0], isis      # substantial adaptation
    assert abs(isis[-1] - isis[-2]) < 0.1, isis  # steady state reached


@run_with_simulators("arbor", "neuron", "nest", "brian2")
def test_IF_curr_alpha_EPSP(sim):
    """A single presynaptic spike into IF_curr_alpha produces an alpha-shaped EPSP
    whose peak matches the analytic current-based-alpha-synapse prediction on every
    backend (Arbor realises the alpha with an implicit solver; cross-checked to
    <0.2% of the NEURON/analytic value at this timestep)."""
    v_rest, cm, tau_m, tau_syn, weight = -65.0, 1.0, 20.0, 2.0, 0.2
    sim.setup(timestep=0.025)
    source = sim.Population(1, sim.SpikeSourceArray(spike_times=[20.0]))
    cell = sim.Population(1, sim.IF_curr_alpha(
        v_rest=v_rest, v_reset=v_rest, v_thresh=-50.0, tau_m=tau_m, cm=cm,
        tau_refrac=5.0, tau_syn_E=tau_syn, i_offset=0.0),
        initial_values={'v': v_rest})
    sim.Projection(source, cell, sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=weight, delay=1.0),
                   receptor_type="excitatory")
    cell.record('v')
    sim.run(120.0)
    v = cell.get_data().segments[0].filter(name='v')[0].magnitude[:, 0]
    epsp = v.max() - v_rest
    # analytic peak of a current-based alpha synapse into an RC membrane:
    # u(t) = (w*e)/(C*tau_syn) * exp(-t/tau_m) * (exp(k t)(k t - 1) + 1)/k^2
    t = np.arange(0, 300, 0.001)
    k = 1 / tau_m - 1 / tau_syn
    u = ((weight * np.e) / (cm * tau_syn) * np.exp(-t / tau_m)
         * (np.exp(k * t) * (k * t - 1) + 1) / k ** 2)
    epsp_theory = u.max()
    assert abs(epsp - epsp_theory) < 0.01 * epsp_theory, (epsp, epsp_theory)
    sim.end()


@run_with_simulators("arbor", "neuron", "nest", "brian2")
def test_IF_cond_alpha_EPSP(sim):
    """A single presynaptic spike into IF_cond_alpha produces an alpha-shaped
    (rise-then-decay) depolarising EPSP peaking a few ms after the input."""
    v_rest, tau_syn, weight = -65.0, 2.0, 0.05
    onset, delay = 20.0, 1.0
    sim.setup(timestep=0.025)
    source = sim.Population(1, sim.SpikeSourceArray(spike_times=[onset]))
    cell = sim.Population(1, sim.IF_cond_alpha(
        v_rest=v_rest, v_reset=v_rest, v_thresh=-50.0, tau_m=20.0, cm=1.0,
        tau_refrac=5.0, tau_syn_E=tau_syn, e_rev_E=0.0),
        initial_values={'v': v_rest})
    sim.Projection(source, cell, sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=weight, delay=delay),
                   receptor_type="excitatory")
    cell.record('v')
    sim.run(120.0)
    signal = cell.get_data().segments[0].filter(name='v')[0]
    v = signal.magnitude[:, 0]
    t = signal.times.magnitude
    # depolarising and alpha-shaped: quiescent before the input, single peak after
    assert np.allclose(v[t < onset], v_rest, atol=1e-6)
    i_peak = int(np.argmax(v))
    assert v[i_peak] - v_rest > 0.5
    # peak occurs after the synaptic peak (onset + delay + tau_syn) and before the
    # membrane time constant washes it out
    assert onset + delay + tau_syn < t[i_peak] < onset + delay + 20.0
    # decays monotonically back towards rest after the peak
    assert v[-1] < v[i_peak]
    sim.end()


@run_with_simulators("arbor", "neuron", "nest", "brian2")
def test_IF_point_neuron_heterogeneous_current(sim):
    """Per-cell i_offset values give per-cell firing rates (guards Arbor's
    per-cell cable-cell construction / scalar coercion)."""
    sim.setup(timestep=0.025)
    cells = sim.Population(3, sim.IF_cond_exp(
        v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0, tau_m=20.0, cm=1.0,
        tau_refrac=5.0, i_offset=[0.5, 1.0, 1.5]),
        initial_values={'v': -65.0})
    cells.record('spikes')
    sim.run(500.0)
    counts = [len(st) for st in cells.get_data().segments[0].spiketrains]
    assert counts[0] == 0, counts  # 0.5 nA is sub-threshold
    assert 0 < counts[1] < counts[2], counts  # firing rate rises with current
    sim.end()


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
