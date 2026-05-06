import os
import numpy as np
import pytest

try:
    import pyNN.nest as sim
    have_nest = True
except ImportError:
    have_nest = False

try:
    import pynestml  # noqa: F401  # pip install name is "nestml", import name is "pynestml"
    have_pynestml = True
except ImportError:
    have_pynestml = False

NESTML_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "nestml")


def _nestml_ver():
    try:
        import pynestml
        parts = pynestml.__version__.split('.')
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (0, 0)


# NESTML 8.2 syntax: output: spike(weight real, delay ms), emit_spike(w, d)
_STDP_SYNAPSE_82 = """\
model stdp_synapse_82:
    state:
        w real = 1 [[w >= 0]]
        pre_trace real = 0.
        post_trace real = 0.

    parameters:
        d ms = 1 ms
        lambda real = 0.01
        alpha real = 1
        tau_tr_pre ms = 20 ms
        tau_tr_post ms = 20 ms
        mu_plus real = 1
        mu_minus real = 1
        Wmin real = 0. [[Wmin >= 0]]
        Wmax real = 100. [[Wmax >= 0]]

    equations:
        pre_trace' = -pre_trace / tau_tr_pre
        post_trace' = -post_trace / tau_tr_post

    input:
        pre_spikes <- spike
        post_spikes <- spike

    output:
        spike(weight real, delay ms)

    onReceive(post_spikes):
        post_trace += 1
        w_ real = Wmax * (w / Wmax + (lambda * (1. - (w / Wmax))**mu_plus * pre_trace))
        w = min(Wmax, w_)

    onReceive(pre_spikes):
        pre_trace += 1
        w_ real = Wmax * (w / Wmax - (alpha * lambda * (w / Wmax)**mu_minus * post_trace))
        w = max(Wmin, w_)
        emit_spike(w, d)

    update:
        integrate_odes()
"""

_TSODYKS_SYNAPSE_82 = """\
model tsodyks_synapse_82_nestml:
    parameters:
        w real = 1
        d ms = 1 ms
        tau_psc ms = 3 ms
        tau_fac ms = 0 ms
        tau_rec ms = 800 ms
        U real = 0.5

    state:
        x real = 1
        y real = 0
        u real = 0.5
        t_last_update ms = 0 ms

    input:
        pre_spikes <- spike

    output:
        spike(weight real, delay ms)

    onReceive(pre_spikes):
        dt ms = t - t_last_update
        t_last_update = t

        Puu real = tau_fac == 0 ms ? 0 : exp(-dt / tau_fac)
        Pyy real = exp(-dt / tau_psc)
        Pzz real = exp(-dt / tau_rec)
        Pxy real = ((Pzz - 1) * tau_rec - (Pyy - 1) * tau_psc) / (tau_psc - tau_rec)
        Pxz real = 1 - Pzz
        z real = 1 - x - y

        u *= Puu
        x += Pxy * y + Pxz * z
        y *= Pyy
        u += U * (1 - u)

        delta_y_tsp real = u * x
        x -= delta_y_tsp
        y += delta_y_tsp

        emit_spike(delta_y_tsp * w, d)
"""


@pytest.fixture(autouse=True)
def reset_nestml_state():
    """Reset NESTML module-level state before and after each test.

    Resets both before (so state left by non-NESTML tests in the same worker doesn't bleed
    in) and after (so state left by this test doesn't bleed into subsequent tests).
    """
    if have_nest:
        from pyNN.nest import nestml
        nestml._compiled = False
        nestml._pending.clear()
    yield
    if have_nest:
        from pyNN.nest import nestml
        nestml._compiled = False
        nestml._pending.clear()


def test_nestml_cell_type_vm_trace():
    """NESTML iaf_psc_exp_neuron V_m trace should be numerically identical to native NEST iaf_psc_exp."""
    if not have_nest:
        pytest.skip("nest not available")
    if not have_pynestml:
        pytest.skip("pynestml not available")

    from pyNN.nest import nestml as pynn_nestml
    iaf_path = os.path.join(NESTML_MODEL_DIR, "iaf_psc_exp_neuron.nestml")
    NestmlIAF = pynn_nestml.nestml_cell_type("iaf_psc_exp_neuron", iaf_path)

    sim.setup(timestep=0.1, min_delay=1.0)

    nestml_pop = sim.Population(1, NestmlIAF(I_e=400.0), label="nestml")
    native_pop = sim.Population(1, sim.native_cell_type("iaf_psc_exp")(I_e=400.0), label="native")
    nestml_pop.record("V_m")
    native_pop.record("V_m")

    sim.run(100.0)

    nestml_vm = nestml_pop.get_data().segments[0].filter(name="V_m")[0].magnitude
    native_vm = native_pop.get_data().segments[0].filter(name="V_m")[0].magnitude

    assert nestml_vm.shape == native_vm.shape
    assert np.ptp(native_vm) > 5.0, "native iaf_psc_exp shows no dynamics — check I_e"
    np.testing.assert_allclose(nestml_vm, native_vm, atol=1e-9,
                               err_msg="V_m traces differ between NESTML and native iaf_psc_exp")

    sim.end()


@pytest.mark.skipif(not have_pynestml or _nestml_ver() < (8, 3),
                    reason="requires NESTML >= 8.3")
def test_nestml_synapse_weight_changes():
    """STDP synapse weights should change from their initial value after Poisson-driven activity."""
    if not have_nest:
        pytest.skip("nest not available")

    from pyNN.nest import nestml as pynn_nestml
    iaf_path = os.path.join(NESTML_MODEL_DIR, "iaf_psc_exp_neuron.nestml")
    stdp_path = os.path.join(NESTML_MODEL_DIR, "stdp_synapse.nestml")
    stdp_cls = pynn_nestml.nestml_synapse_type(
        "stdp_synapse", stdp_path,
        postsynaptic_neuron_nestml_description=iaf_path,
    )
    PostCellType = stdp_cls.postsynaptic_cell_type

    sim.setup(timestep=0.1, min_delay=1.0)

    source = sim.Population(10, sim.SpikeSourcePoisson(rate=100.0), label="source")
    target = sim.Population(10, PostCellType(), label="target")

    initial_weight = 1.0
    prj = sim.Projection(
        source, target,
        sim.AllToAllConnector(),
        stdp_cls(weight=initial_weight, delay=1.0),
        receptor_type="excitatory",
    )

    sim.run(1000.0)

    weights = np.array(prj.get("weight", format="list"))[:, 2]
    assert not np.allclose(weights, initial_weight), \
        "STDP weights did not change from initial value — plasticity may not be active"

    sim.end()


@pytest.mark.skipif(not have_pynestml or _nestml_ver() < (8, 3),
                    reason="requires NESTML >= 8.3")
def test_nestml_tsodyks_synapse_vm_trace():
    """NESTML tsodyks_synapse postsynaptic V_m should be numerically identical to native NEST tsodyks_synapse."""
    if not have_nest:
        pytest.skip("nest not available")

    from pyNN.nest import nestml as pynn_nestml
    tsodyks_path = os.path.join(NESTML_MODEL_DIR, "tsodyks_synapse.nestml")
    TsodyksSyn = pynn_nestml.nestml_synapse_type(
        "tsodyks_synapse_nestml", tsodyks_path,
        weight_variable="w",
    )

    sim.setup(timestep=0.1, min_delay=1.0)

    spike_times = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]
    source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times), label="source")

    nestml_target = sim.Population(1, sim.native_cell_type("iaf_psc_exp")(), label="nestml_target")
    native_target = sim.Population(1, sim.native_cell_type("iaf_psc_exp")(), label="native_target")

    NativeTsodyks = sim.native_synapse_type("tsodyks_synapse")

    sim.Projection(
        source, nestml_target,
        sim.AllToAllConnector(),
        TsodyksSyn(weight=500.0, delay=1.0),
        receptor_type="excitatory",
    )
    sim.Projection(
        source, native_target,
        sim.AllToAllConnector(),
        NativeTsodyks(weight=500.0, delay=1.0, tau_psc=3.0, tau_fac=0.0, tau_rec=800.0, U=0.5),
        receptor_type="excitatory",
    )

    nestml_target.record("V_m")
    native_target.record("V_m")

    sim.run(500.0)

    nestml_vm = nestml_target.get_data().segments[0].filter(name="V_m")[0].magnitude
    native_vm = native_target.get_data().segments[0].filter(name="V_m")[0].magnitude

    assert nestml_vm.shape == native_vm.shape
    assert np.ptp(native_vm) > 1.0, "native tsodyks_synapse target shows no response — check weight/spike_times"
    np.testing.assert_allclose(nestml_vm, native_vm, atol=1e-6,
                               err_msg="V_m traces differ between NESTML and native tsodyks_synapse")

    sim.end()


def test_nestml_cell_type_inline_string():
    """nestml_cell_type() should accept inline NESTML source, not just file paths."""
    if not have_nest:
        pytest.skip("nest not available")
    if not have_pynestml:
        pytest.skip("pynestml not available")

    from pyNN.nest import nestml as pynn_nestml
    iaf_path = os.path.join(NESTML_MODEL_DIR, "iaf_psc_exp_neuron.nestml")
    with open(iaf_path) as f:
        iaf_source = f.read()

    NestmlIAF = pynn_nestml.nestml_cell_type("iaf_psc_exp_neuron", iaf_source)
    sim.setup(timestep=0.1, min_delay=1.0)

    pop = sim.Population(1, NestmlIAF(I_e=400.0), label="nestml_inline")
    pop.record("V_m")
    sim.run(100.0)

    vm = pop.get_data().segments[0].filter(name="V_m")[0].magnitude
    assert np.ptp(vm) > 5.0, "No membrane potential dynamics — inline model may not have compiled"

    sim.end()


def test_nestml_register_after_setup_raises():
    """Calling nestml_cell_type() after sim.setup() should raise RuntimeError."""
    if not have_nest:
        pytest.skip("nest not available")

    from pyNN.nest import nestml as pynn_nestml
    sim.setup(timestep=0.1, min_delay=1.0)
    iaf_path = os.path.join(NESTML_MODEL_DIR, "iaf_psc_exp_neuron.nestml")

    with pytest.raises(RuntimeError, match="before sim.setup"):
        pynn_nestml.nestml_cell_type("iaf_psc_exp_neuron", iaf_path)

    sim.end()


def test_nestml_setup_without_models():
    """sim.setup() should succeed and set _compiled even when no NESTML models are registered."""
    if not have_nest:
        pytest.skip("nest not available")

    from pyNN.nest import nestml as pynn_nestml
    sim.setup(timestep=0.1, min_delay=1.0)

    assert pynn_nestml._compiled, "_compiled should be True after setup() even with no models"
    assert pynn_nestml._pending == [], "_pending should be empty after setup()"

    sim.end()


@pytest.mark.skipif(not have_pynestml or _nestml_ver() >= (8, 3),
                    reason="requires NESTML < 8.3")
def test_nestml_synapse_weight_changes_82():
    """STDP synapse with NESTML 8.2 syntax (delay in emit_spike) works on NESTML 8.2."""
    if not have_nest:
        pytest.skip("nest not available")

    from pyNN.nest import nestml as pynn_nestml
    iaf_path = os.path.join(NESTML_MODEL_DIR, "iaf_psc_exp_neuron.nestml")
    stdp_cls = pynn_nestml.nestml_synapse_type(
        "stdp_synapse_82", _STDP_SYNAPSE_82,
        postsynaptic_neuron_nestml_description=iaf_path,
        delay_variable="d",
    )
    PostCellType = stdp_cls.postsynaptic_cell_type

    sim.setup(timestep=0.1, min_delay=1.0)

    source = sim.Population(10, sim.SpikeSourcePoisson(rate=100.0), label="source")
    target = sim.Population(10, PostCellType(), label="target")

    initial_weight = 1.0
    prj = sim.Projection(
        source, target,
        sim.AllToAllConnector(),
        stdp_cls(weight=initial_weight, delay=1.0),
        receptor_type="excitatory",
    )

    sim.run(1000.0)

    weights = np.array(prj.get("weight", format="list"))[:, 2]
    assert not np.allclose(weights, initial_weight), \
        "STDP weights did not change from initial value — plasticity may not be active"

    sim.end()


@pytest.mark.skipif(not have_pynestml or _nestml_ver() >= (8, 3),
                    reason="requires NESTML < 8.3")
def test_nestml_tsodyks_synapse_vm_trace_82():
    """NESTML tsodyks synapse (NESTML 8.2 syntax) V_m matches native NEST tsodyks_synapse."""
    if not have_nest:
        pytest.skip("nest not available")

    from pyNN.nest import nestml as pynn_nestml
    TsodyksSyn = pynn_nestml.nestml_synapse_type(
        "tsodyks_synapse_82_nestml", _TSODYKS_SYNAPSE_82,
        weight_variable="w",
        delay_variable="d",
    )

    sim.setup(timestep=0.1, min_delay=1.0)

    spike_times = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]
    source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times), label="source")

    nestml_target = sim.Population(1, sim.native_cell_type("iaf_psc_exp")(), label="nestml_target")
    native_target = sim.Population(1, sim.native_cell_type("iaf_psc_exp")(), label="native_target")

    NativeTsodyks = sim.native_synapse_type("tsodyks_synapse")

    sim.Projection(
        source, nestml_target,
        sim.AllToAllConnector(),
        TsodyksSyn(weight=500.0, delay=1.0),
        receptor_type="excitatory",
    )
    sim.Projection(
        source, native_target,
        sim.AllToAllConnector(),
        NativeTsodyks(weight=500.0, delay=1.0, tau_psc=3.0, tau_fac=0.0, tau_rec=800.0, U=0.5),
        receptor_type="excitatory",
    )

    nestml_target.record("V_m")
    native_target.record("V_m")

    sim.run(500.0)

    nestml_vm = nestml_target.get_data().segments[0].filter(name="V_m")[0].magnitude
    native_vm = native_target.get_data().segments[0].filter(name="V_m")[0].magnitude

    assert nestml_vm.shape == native_vm.shape
    assert np.ptp(native_vm) > 1.0, "native tsodyks_synapse target shows no response"
    np.testing.assert_allclose(nestml_vm, native_vm, atol=1e-6,
                               err_msg="V_m traces differ between NESTML 8.2 and native tsodyks_synapse")

    sim.end()


def test_nestml_wrong_syntax_raises():
    """Wrong NESTML synapse syntax for the installed version raises an informative RuntimeError."""
    if not have_nest:
        pytest.skip("nest not available")
    if not have_pynestml:
        pytest.skip("pynestml not available")

    from pyNN.nest import nestml as pynn_nestml
    stdp_path = os.path.join(NESTML_MODEL_DIR, "stdp_synapse.nestml")

    if _nestml_ver() < (8, 3):
        # NESTML 8.2 installed: new-syntax model (no delay_variable) should raise
        pynn_nestml.nestml_synapse_type("stdp_synapse", stdp_path)
        with pytest.raises(RuntimeError, match="delay_variable"):
            sim.setup(timestep=0.1, min_delay=1.0)
    else:
        # NESTML 8.3+ installed: old-syntax model (with delay_variable) should raise
        pynn_nestml.nestml_synapse_type("stdp_synapse_82", _STDP_SYNAPSE_82,
                                        delay_variable="d")
        with pytest.raises(RuntimeError, match="NESTML model compilation failed silently"):
            sim.setup(timestep=0.1, min_delay=1.0)
