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

    def test_electrode_param_units(self):
        units = arbor_cells.ELECTRODE_PARAM_UNITS
        self.assertIs(units["tstart"], U.ms)
        self.assertIs(units["duration"], U.ms)
        self.assertIs(units["current"], U.nA)


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
