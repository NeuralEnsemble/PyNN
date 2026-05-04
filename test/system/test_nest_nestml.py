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


def test_nestml_synapse_weight_changes():
    """STDP synapse weights should change from their initial value after Poisson-driven activity."""
    if not have_nest:
        pytest.skip("nest not available")
    if not have_pynestml:
        pytest.skip("pynestml not available")

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


def test_nestml_tsodyks_synapse_vm_trace():
    """NESTML tsodyks_synapse postsynaptic V_m should be numerically identical to native NEST tsodyks_synapse."""
    if not have_nest:
        pytest.skip("nest not available")
    if not have_pynestml:
        pytest.skip("pynestml not available")

    from pyNN.nest import nestml as pynn_nestml
    tsodyks_path = os.path.join(NESTML_MODEL_DIR, "tsodyks_synapse.nestml")
    TsodyksSyn = pynn_nestml.nestml_synapse_type(
        "tsodyks_synapse_nestml", tsodyks_path,
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
