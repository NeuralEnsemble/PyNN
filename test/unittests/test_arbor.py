"""
Unit tests for the Arbor backend.

These require Arbor to be installed (they exercise the backend's translation,
compatibility-shim, morphology and catalogue logic against the real ``arbor``
package), but do not run a full simulation. They are skipped when Arbor is not
available.

Several of these are regression guards for bugs found while adding support for
Arbor 0.10.0 - 0.12.2 (see pyNN/arbor/_compat.py).
"""

import importlib
import unittest

try:
    import arbor
    from arbor import units as U
    import pyNN.arbor  # noqa: F401  (ensures the backend imports / catalogue builds)
    have_arbor = True
except ImportError:
    have_arbor = False

# The submodule names `cells` and `morphology` are shadowed by re-exports in
# pyNN/arbor/__init__.py, so import the real submodules explicitly.
if have_arbor:
    _compat = importlib.import_module("pyNN.arbor._compat")
    arbor_cells = importlib.import_module("pyNN.arbor.cells")
    arbor_morphology = importlib.import_module("pyNN.arbor.morphology")
    arbor_simulator = importlib.import_module("pyNN.arbor.simulator")
    arbor_standardmodels = importlib.import_module("pyNN.arbor.standardmodels")


@unittest.skipUnless(have_arbor, "Requires Arbor")
class TestCompatShims(unittest.TestCase):
    """Capability-based shims that let one codebase support Arbor 0.10 - 0.12."""

    def test_get_electrode_mechanism_handles_rename(self):
        # iclamp was renamed to i_clamp in Arbor 0.12.
        mech = _compat.get_electrode_mechanism("iclamp")
        self.assertIn(mech.__name__, ("iclamp", "i_clamp"))
        # constructible with unit-typed args (tstart, duration, current)
        mech(0.0 * U.ms, 1.0 * U.ms, 0.1 * U.nA)

    def test_get_electrode_mechanism_unknown(self):
        with self.assertRaises(AttributeError):
            _compat.get_electrode_mechanism("not_a_mechanism")

    def test_max_extent_policy(self):
        # returns a valid cv_policy whether or not the installed version wants units
        policy = _compat.max_extent_policy(10.0)
        self.assertIsInstance(policy, arbor.cv_policy)

    def test_make_cable_cell(self):
        tree = arbor.segment_tree()
        tree.append(arbor.mnpos, arbor.mpoint(0, 0, 0, 5),
                    arbor.mpoint(10, 0, 0, 5), tag=1)
        decor = arbor.decor()
        decor.set_property(Vm=-60 * U.mV)
        labels = arbor.label_dict({})
        policy = _compat.max_extent_policy(10.0)
        cell = _compat.make_cable_cell(tree, decor, labels, policy)
        self.assertIsInstance(cell, arbor.cable_cell)

    def test_place_current_source(self):
        # 0.12 dropped the label arg from the current-stimulus place() overload.
        decor = arbor.decor()
        mech = _compat.get_electrode_mechanism("iclamp")(
            0.0 * U.ms, 1.0 * U.ms, 0.1 * U.nA)
        # should not raise on any supported version
        _compat.place_current_source(decor, "(root)", mech, "stim_label")


@unittest.skipUnless(have_arbor, "Requires Arbor")
class TestUnitMaps(unittest.TestCase):
    """Regression guards for the unit maps used when translating to Arbor."""

    def test_poisson_schedule_freq_is_Hz_not_kHz(self):
        # PyNN's rate is in Hz; tagging it as kHz gave 1000x too many spikes.
        units = arbor_standardmodels.SpikeSourcePoisson.arbor_schedule_units
        self.assertIs(units["freq"], U.Hz)
        self.assertIs(units["tstart"], U.ms)
        self.assertIs(units["tstop"], U.ms)

    def test_spike_source_array_units(self):
        units = arbor_standardmodels.SpikeSourceArray.arbor_schedule_units
        self.assertEqual(list(units.keys()), ["times"])
        self.assertIs(units["times"], U.ms)

    def test_if_curr_delta_lif_param_units(self):
        units = arbor_standardmodels.IF_curr_delta.lif_param_units
        self.assertIs(units["E_L"], U.mV)
        self.assertIs(units["E_R"], U.mV)
        self.assertIs(units["V_th"], U.mV)
        self.assertIs(units["t_ref"], U.ms)
        self.assertIs(units["tau_m"], U.ms)
        self.assertIs(units["C_m"], U.nF)


@unittest.skipUnless(have_arbor, "Requires Arbor")
class TestTranslations(unittest.TestCase):
    """Standard-model parameter translation (PyNN name/units -> Arbor)."""

    def test_dcsource_translation(self):
        m = arbor_standardmodels.DCSource(amplitude=0.5, start=10.0, stop=20.0)
        native = m.native_parameters
        native.shape = (1,)
        native.evaluate(simplify=True)
        d = native.as_dict()
        self.assertAlmostEqual(d["current"], 0.5)
        self.assertAlmostEqual(d["tstart"], 10.0)
        self.assertAlmostEqual(d["duration"], 10.0)  # stop - start

    def test_spike_source_poisson_translation(self):
        t = arbor_standardmodels.SpikeSourcePoisson.translations
        self.assertEqual(t["rate"]["translated_name"], "freq")
        self.assertEqual(t["start"]["translated_name"], "tstart")
        self.assertEqual(t["duration"]["translated_name"], "tstop")
        # duration -> tstop = start + duration (stored as a string expression)
        self.assertEqual(t["duration"]["forward_transform"], "start + duration")

    def test_spike_source_array_translation(self):
        t = arbor_standardmodels.SpikeSourceArray.translations
        self.assertEqual(t["spike_times"]["translated_name"], "times")

    def test_multicompartment_schema(self):
        schema = arbor_standardmodels.MultiCompartmentNeuron().get_schema()
        for key in ("morphology", "cm", "Ra", "ionic_species"):
            self.assertIn(key, schema)

    def test_if_curr_delta_is_native_lif(self):
        self.assertIs(arbor_standardmodels.IF_curr_delta.arbor_cell_kind,
                      arbor.cell_kind.lif)

    def test_if_curr_delta_translation(self):
        m = arbor_standardmodels.IF_curr_delta(
            v_rest=-65.0, cm=1.0, tau_m=20.0, tau_refrac=2.0,
            v_reset=-70.0, v_thresh=-50.0, i_offset=0.0)
        native = m.native_parameters
        native.shape = (1,)
        native.evaluate(simplify=True)
        d = native.as_dict()
        self.assertAlmostEqual(d["E_L"], -65.0)
        self.assertAlmostEqual(d["E_R"], -70.0)
        self.assertAlmostEqual(d["V_th"], -50.0)
        self.assertAlmostEqual(d["t_ref"], 2.0)
        self.assertAlmostEqual(d["tau_m"], 20.0)
        self.assertAlmostEqual(d["C_m"], 1.0)  # cm passes through in nF
        self.assertAlmostEqual(d["i_offset"], 0.0)


@unittest.skipUnless(have_arbor, "Requires Arbor")
class TestPointNeurons(unittest.TestCase):
    """Point neurons realised as single-compartment cable cells (Phase 2)."""

    def test_lif_translation(self):
        m = arbor_standardmodels.LIF(
            v_rest=-65.0, cm=1.0, tau_m=20.0, tau_refrac=2.0,
            v_reset=-70.0, v_thresh=-50.0, i_offset=0.1)
        native = m.native_parameters
        native.shape = (1,)
        native.evaluate(simplify=True)
        d = native.as_dict()
        self.assertAlmostEqual(d["E_L"], -65.0)
        self.assertAlmostEqual(d["E_R"], -70.0)
        self.assertAlmostEqual(d["V_th"], -50.0)
        self.assertAlmostEqual(d["t_ref"], 2.0)
        self.assertAlmostEqual(d["tau_m"], 20.0)
        self.assertAlmostEqual(d["C_m"], 1.0)  # cm passes through in nF
        self.assertAlmostEqual(d["i_offset"], 0.1)

    def test_curr_exp_psr(self):
        psr = arbor_standardmodels.CurrExpPostSynapticResponse(tau_syn=3.0)
        self.assertEqual(psr.model, "expsyn_curr")
        self.assertFalse(psr.conductance_based)
        native = psr.native_parameters
        native.shape = (1,)
        native.evaluate(simplify=True)
        self.assertAlmostEqual(native.as_dict()["tau"], 3.0)

    def test_cond_exp_psr(self):
        psr = arbor_standardmodels.CondExpPostSynapticResponse(tau_syn=3.0, e_syn=-10.0)
        self.assertEqual(psr.model, "expsyn")
        self.assertTrue(psr.conductance_based)
        native = psr.native_parameters
        native.shape = (1,)
        native.evaluate(simplify=True)
        d = native.as_dict()
        self.assertAlmostEqual(d["tau"], 3.0)
        self.assertAlmostEqual(d["e"], -10.0)

    def test_point_neuron_is_cable_cell(self):
        ct = arbor_standardmodels.PointNeuron(
            arbor_standardmodels.LIF(),
            excitatory=arbor_standardmodels.CondExpPostSynapticResponse(),
            inhibitory=arbor_standardmodels.CondExpPostSynapticResponse(e_syn=-70.0),
        )
        self.assertIs(ct.arbor_cell_kind, arbor.cell_kind.cable)
        self.assertEqual(ct.receptor_types, ["excitatory", "inhibitory"])
        self.assertTrue(ct.conductance_based)

    def test_point_neuron_mixed_receptor_kinds_rejected(self):
        ct = arbor_standardmodels.PointNeuron(
            arbor_standardmodels.LIF(),
            excitatory=arbor_standardmodels.CondExpPostSynapticResponse(),
            inhibitory=arbor_standardmodels.CurrExpPostSynapticResponse(),
        )
        with self.assertRaises(Exception):
            ct.conductance_based

    def test_if_cond_exp_is_cable_cell(self):
        ct = arbor_standardmodels.IF_cond_exp(tau_syn_E=1.5, e_rev_E=0.0)
        self.assertIs(ct.arbor_cell_kind, arbor.cell_kind.cable)
        self.assertTrue(ct.conductance_based)
        self.assertEqual(tuple(ct.receptor_types), ("excitatory", "inhibitory"))

    def test_if_curr_exp_is_cable_cell(self):
        ct = arbor_standardmodels.IF_curr_exp(tau_syn_E=1.5)
        self.assertIs(ct.arbor_cell_kind, arbor.cell_kind.cable)
        self.assertFalse(ct.conductance_based)

    def test_if_curr_exp_translation(self):
        # the classic model uses the standard flat translate(); check a couple of
        # translated native names appear in the resulting synapse parameters
        ct = arbor_standardmodels.IF_curr_exp(tau_syn_E=1.5, tau_syn_I=2.5, cm=0.5)
        builder = ct.native_parameters["cell_description"].base_value
        builder.set_shape((1,))
        exc_model, exc_params = builder.post_synaptic_receptors["excitatory"]
        self.assertEqual(exc_model, "expsyn_curr")
        self.assertAlmostEqual(exc_params["tau"][0], 1.5)
        self.assertAlmostEqual(builder.neuron_parameters["C_m"][0], 0.5)

    def test_reset_and_current_synapse_mechanisms_in_catalogue(self):
        cat = arbor.load_catalogue(arbor_simulator.catalogue_path())
        mechs = list(cat)
        self.assertIn("lif", mechs)
        self.assertIn("expsyn_curr", mechs)


@unittest.skipUnless(have_arbor, "Requires Arbor")
class TestCurrentSources(unittest.TestCase):
    """Each standard current source is realised as one or more Arbor iclamp
    (envelope, frequency_Hz, phase_deg) components; check they are built correctly."""

    @staticmethod
    def _components(source):
        source.parameter_space.shape = (1,)
        return source._iclamp_components()

    def test_dcsource_is_a_box(self):
        components = self._components(
            arbor_standardmodels.DCSource(amplitude=0.5, start=10.0, stop=20.0))
        self.assertEqual(len(components), 1)
        envelope, frequency, phase = components[0]
        self.assertEqual(frequency, 0.0)
        # rectangular pulse: on at start, off at stop
        self.assertEqual(envelope, [(10.0, 0.5), (20.0, 0.5), (20.0, 0.0)])

    def test_stepcurrentsource_is_a_staircase(self):
        components = self._components(arbor_standardmodels.StepCurrentSource(
            times=[10.0, 15.0, 20.0], amplitudes=[0.1, 0.2, 0.3]))
        self.assertEqual(len(components), 1)
        envelope, frequency, _ = components[0]
        self.assertEqual(frequency, 0.0)
        # piecewise-constant with duplicated breakpoints; holds the last value
        self.assertEqual(
            [(round(t, 6), round(a, 6)) for (t, a) in envelope],
            [(10.0, 0.1), (15.0, 0.1), (15.0, 0.2), (20.0, 0.2), (20.0, 0.3)])

    def test_stepcurrentsource_rejects_bad_times(self):
        with self.assertRaises(ValueError):
            self._components(arbor_standardmodels.StepCurrentSource(
                times=[10.0, -5.0], amplitudes=[0.1, 0.2]))
        with self.assertRaises(ValueError):
            self._components(arbor_standardmodels.StepCurrentSource(
                times=[10.0, 5.0], amplitudes=[0.1, 0.2]))

    def test_acsource_sine_plus_offset(self):
        components = self._components(arbor_standardmodels.ACSource(
            start=10.0, stop=20.0, amplitude=0.5, offset=0.1,
            frequency=100.0, phase=30.0))
        self.assertEqual(len(components), 2)
        (sine_env, freq, phase), (offset_env, offset_freq, _) = components
        self.assertEqual(freq, 100.0)
        # phase shifted so PyNN's 30 deg holds at start=10 (f=100 Hz -> 360 deg/10 ms)
        self.assertAlmostEqual(phase, 30.0 - 360.0 * 100.0 * 10.0 / 1000.0)
        self.assertEqual(sine_env, [(10.0, 0.5), (20.0, 0.5), (20.0, 0.0)])
        self.assertEqual(offset_freq, 0.0)
        self.assertEqual(offset_env, [(10.0, 0.1), (20.0, 0.1), (20.0, 0.0)])

    def test_acsource_omits_zero_offset(self):
        components = self._components(arbor_standardmodels.ACSource(
            start=10.0, stop=20.0, amplitude=0.5, offset=0.0,
            frequency=100.0, phase=0.0))
        self.assertEqual(len(components), 1)

    def test_noisycurrentsource_samples_and_zeroes_at_stop(self):
        import pyNN.arbor as sim
        sim.setup(timestep=0.1, min_delay=0.1)
        components = self._components(arbor_standardmodels.NoisyCurrentSource(
            mean=0.5, stdev=0.05, start=10.0, stop=12.0, dt=0.5))
        self.assertEqual(len(components), 1)
        envelope, frequency, _ = components[0]
        self.assertEqual(frequency, 0.0)
        self.assertAlmostEqual(envelope[0][0], 10.0)   # starts at `start`
        self.assertAlmostEqual(envelope[-1][0], 12.0)  # ends at `stop`
        self.assertEqual(envelope[-1][1], 0.0)         # current off at `stop`

    def test_noisycurrentsource_requires_finite_stop(self):
        import pyNN.arbor as sim
        sim.setup(timestep=0.1, min_delay=0.1)
        with self.assertRaises(ValueError):
            self._components(arbor_standardmodels.NoisyCurrentSource(
                mean=0.5, stdev=0.05, start=10.0))


@unittest.skipUnless(have_arbor, "Requires Arbor")
class TestMorphologyLocsets(unittest.TestCase):
    """Locset generation for recording/placement locations."""

    def test_labelled_locations_use_direct_locsets(self):
        # Regression guard: Arbor 0.12's probe-label resolution does not resolve
        # label references in probe locsets, so we must emit resolved expressions.
        gen = arbor_morphology.LabelledLocations("soma", "dendrite")
        locsets = gen.generate_locations(morphology=None, label="rec")
        by_name = {label.rsplit("-", 1)[-1]: locset for locset, label in locsets}
        self.assertEqual(by_name["soma"], "(root)")
        self.assertEqual(by_name["dendrite"], "(location 0 0.5)")
        # must be resolved expressions, not label references
        for locset, _ in locsets:
            self.assertFalse(locset.startswith('"'))


@unittest.skipUnless(have_arbor, "Requires Arbor")
class TestSimulatorHelpers(unittest.TestCase):

    def test_catalogue_path_is_version_keyed(self):
        path = arbor_simulator.catalogue_path()
        self.assertTrue(path.endswith(f"PyNN-catalogue-{arbor.__version__}.so"), path)

    def test_convert_point(self):
        from neuroml import Point3DWithDiam
        p = Point3DWithDiam(x=1.0, y=2.0, z=3.0, diameter=4.0)
        mp = arbor_cells.convert_point(p)
        self.assertAlmostEqual(mp.x, 1.0)
        self.assertAlmostEqual(mp.y, 2.0)
        self.assertAlmostEqual(mp.z, 3.0)
        self.assertAlmostEqual(mp.radius, 2.0)  # diameter / 2

    def test_region_name_to_tag(self):
        self.assertEqual(arbor_cells.region_name_to_tag("soma"), 1)
        self.assertEqual(arbor_cells.region_name_to_tag("axon"), 2)
        self.assertEqual(arbor_cells.region_name_to_tag("dendrite"), 3)
        self.assertEqual(arbor_cells.region_name_to_tag("unknown"), -1)


if __name__ == "__main__":
    unittest.main()
