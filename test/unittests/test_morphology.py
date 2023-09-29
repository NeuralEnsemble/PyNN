"""
Unit tests of the morphology module

:copyright: Copyright 2018 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import os
import unittest
from math import sqrt
import numpy as np
from numpy.testing import assert_array_equal
try:
    import neuroml
    import neuroml.arraymorph
    from neuroml import Morphology as NMLMorphology, Segment, SegmentGroup, Member, Point3DWithDiam as P
    have_neuroml = True
except ImportError:
    have_neuroml = False
from pyNN.morphology import load_morphology, NeuroMLMorphology, SectionType, any as morph_any
try:
    from pyNN.neuron.morphology import (dendrites, apical_dendrites,
                                 basal_dendrites, random_section, with_label,
                                 uniform, by_diameter)
    have_neuron = True
except ModuleNotFoundError:
    have_neuron = False
import pytest


morph_data = """# test morphology
1 1 0.0 0.0 0.0 10.0 -1
2 3 0.0 -10.0 0.0 1.1 1
3 3 -10.0 -20.0 0.0 1.0 2
4 3 -20.0 -30.0 0.0 0.9 3
5 3 0.0 -10.0 0.0 1.1 1
6 3 10.0 -20.0 0.0 1.0 5
7 3 20.0 -30.0 0.0 0.9 6
8 4 0.0 10.0 0.0 1.5 1
9 4 0.0 20.0 0.0 1.4 8
10 4 0.0 30.0 0.0 1.3 9
11 4 0.0 40.0 0.0 1.2 10
12 4 10.0 50.0 0.0 1.0 11
13 4 20.0 60.0 0.0 0.9 12
14 4 -10.0 50.0 0.0 1.0 11
15 4 -20.0 60.0 0.0 0.9 14
16 2 10.0 0.0 0.0 0.3 1
17 2 110.0 0.0 0.0 0.2 16
18 2 210.0 0.0 0.0 0.2 17
"""


def setUpModule():
    global neuroml_morph, array_morph

    if not have_neuroml:
            pytest.skip("libNeuroML not installed")

    test_file = "morph_test.swc"
    with open(test_file, "w") as fp:
        fp.write(morph_data)
    array_morph = load_morphology(test_file)
    #os.remove(test_file)

    soma = Segment(proximal=P(x=0, y=-10, z=0, diameter=20.0),
                       distal=P(x=0, y=10, z=0, diameter=20.0),
                       name="soma", id="0")
    basal_dendrites = {}
    basal_dendrites["basal00"] = \
        Segment(proximal=P(x=0, y=-10, z=0, diameter=2.2),
                distal=P(x=-10, y=-20, z=0, diameter=2.0),
                name="basal00", parent=soma, id="1")
    basal_dendrites["basal01"] = \
        Segment(proximal=P(x=-10, y=-20, z=0, diameter=2.0),
                distal=P(x=-10, y=-30, z=0, diameter=1.8),
                name="basal01", parent=basal_dendrites["basal00"], id="2")
    basal_dendrites["basal10"] = \
        Segment(proximal=P(x=0, y=-10, z=0, diameter=2.2),
                distal=P(x=10, y=-20, z=0, diameter=2.0),
                name="basal10", parent=soma, id="3")
    basal_dendrites["basal11"] = \
        Segment(proximal=P(x=10, y=-20, z=0, diameter=2.0),
                distal=P(x=20, y=-30, z=0, diameter=1.8),
                name="basal11", parent=basal_dendrites["basal10"], id="4")
    apical_dendrites = {}
    apical_dendrites["apical0"] = \
        Segment(proximal=P(x=0, y=10, z=0, diameter=3.0),
                distal=P(x=0, y=20, z=0, diameter=2.8),
                name="apical0", parent=soma, id="5")
    apical_dendrites["apical1"] = \
        Segment(proximal=P(x=0, y=20, z=0, diameter=2.8),
                    distal=P(x=0, y=30, z=0, diameter=2.6),
                    name="apical1", parent=apical_dendrites["apical0"], id="6")
    apical_dendrites["apical2"] = \
        Segment(proximal=P(x=0, y=30, z=0, diameter=2.6),
                distal=P(x=0, y=40, z=0, diameter=2.4),
                name="apical2", parent=apical_dendrites["apical1"], id="7")
    apical_dendrites["apical300"] = \
        Segment(proximal=P(x=0, y=40, z=0, diameter=2.4),
                distal=P(x=10, y=50, z=0, diameter=2.0),
                name="apical300", parent=apical_dendrites["apical2"], id="8")
    apical_dendrites["apical301"] = \
        Segment(proximal=P(x=10, y=50, z=0, diameter=2.0),
                distal=P(x=20, y=60, z=0, diameter=1.8),
                name="apical301", parent=apical_dendrites["apical300"], id="9")
    apical_dendrites["apical310"] = \
        Segment(proximal=P(x=0, y=40, z=0, diameter=2.4),
                distal=P(x=-10, y=50, z=0, diameter=2.0),
                name="apical300", parent=apical_dendrites["apical2"], id="10")
    apical_dendrites["apical311"] = \
        Segment(proximal=P(x=-10, y=50, z=0, diameter=2.0),
                distal=P(x=-20, y=60, z=0, diameter=1.8),
                name="apical301", parent=apical_dendrites["apical310"], id="11")
    axon = {}
    axon["axon0"] = \
        Segment(proximal=P(x=10, y=0, z=0, diameter=0.6),
                distal=P(x=110, y=0, z=0, diameter=0.4),
                name="axon0", parent=soma, id="12")
    axon["axon1"] = \
        Segment(proximal=P(x=110, y=0, z=0, diameter=0.4),
                distal=P(x=210, y=0, z=0, diameter=0.4),
                name="axon0", parent=axon["axon0"], id="13")

    segments = [soma] \
                + list(basal_dendrites.values()) \
                + list(apical_dendrites.values()) \
                + list(axon.values())

    ## Probably the commented out lines are correct NeuroML
    ## this should be fixed in the NeuroMLMorphology constructor
    # segment_groups = [
    #     SegmentGroup(id="soma_group", members=[Member(soma.id)]),
    #     SegmentGroup(id="basal_dendrites",
    #                     members=[Member(seg.id) for seg in basal_dendrites.values()]),
    #     SegmentGroup(id="apical_dendrites",
    #                     members=[Member(seg.id) for seg in apical_dendrites.values()]),
    #     SegmentGroup(id="axon",
    #                     members=[Member(seg.id) for seg in axon.values()]),
    # ]
    segment_groups = [
        SegmentGroup(id="soma_group", members=[soma]),
        SegmentGroup(id="basal_dendrites",
                        members=[seg for seg in basal_dendrites.values()]),
        SegmentGroup(id="apical_dendrites",
                        members=[seg for seg in apical_dendrites.values()]),
        SegmentGroup(id="axon",
                        members=[seg for seg in axon.values()]),
    ]

    neuroml_morph = NeuroMLMorphology(NMLMorphology(segments=segments,
                                                    segment_groups=segment_groups))


class NeuroMLArrayMorphologyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not have_neuroml:
            pytest.skip("libNeuroML not installed")
        cls.morph = array_morph

    #def test_soma_index(self):
    #    self.assertEqual(self.morph.soma_index, ?)

    def test_segments(self):
        segments = self.morph.segments
        self.assertEqual(len(segments), 17)
        self.assertIsInstance(segments, neuroml.arraymorph.SegmentList)
        self.assertIsInstance(segments[0], neuroml.Segment)

    def test_path_lengths(self):
        path_lengths = self.morph.path_lengths
        assert_array_equal(
            path_lengths,
            np.array([
                0.0, 10.0, 10 + sqrt(200), 10 + sqrt(800), 10, 10 + sqrt(200), 10 + sqrt(800),
                10.0, 20.0, 30.0, 40.0, 40 + sqrt(200), 40 + sqrt(800), 40 + sqrt(200), 40 + sqrt(800),
                10.0, 110.0, 210.0
            ]))

    def test_get_distance(self):
        self.assertEqual(self.morph.get_distance(0), 0.0)
        self.assertEqual(self.morph.get_distance(1), 10.0)
        self.assertEqual(self.morph.get_distance(17), 210.0)

    def test_section_groups(self):
        # section_groups contains index arrays for the sections
        assert_array_equal(self.morph.section_groups[SectionType.apical_dendrite],
                           np.arange(7, 15))

    def test_len(self):
        self.assertEqual(len(self.morph), 17)


class NeuroMLStandardMorphologyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.morph = neuroml_morph

    def test_segments(self):
        segments = self.morph.segments
        self.assertEqual(len(segments), 14)
        self.assertIsInstance(segments, list)
        self.assertIsInstance(segments[0], neuroml.Segment)

    def test_len(self):
        self.assertEqual(len(self.morph), 14)


class MorphologyFilterTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not have_neuron:
            pytest.skip("NEURON not available")
        cls.array_morph = array_morph
        cls.neuroml_morph = neuroml_morph

    def test_dendrites_filter(self):
        filter = dendrites()
        ids = filter(self.array_morph)
        self.assertEqual(len(ids), 14)

    def test_apical_dendrites_filter(self):
        filter = apical_dendrites()
        ids = filter(self.array_morph)
        self.assertEqual(len(ids), 8)

    def test_basal_dendrites_filter(self):
        filter = basal_dendrites()
        ids = filter(self.array_morph)
        self.assertEqual(len(ids), 6)

    def test_random_sample_filter(self):
        filter = random_section(apical_dendrites())
        id = filter(self.array_morph)
        self.assertIn(id, apical_dendrites()(self.array_morph))

    def test_with_label_filter_arraymorph(self):
        filter1 = with_label(SectionType.apical_dendrite)
        filter2 = apical_dendrites()
        index1 = filter1(self.array_morph)
        index2 = filter2(self.array_morph)
        assert_array_equal(index1, index2)

    def test_with_label_filter_neuromlmorph_group(self):
        filter = with_label("apical_dendrites")
        index = filter(self.neuroml_morph)
        segments = [self.neuroml_morph.segments[i] for i in index]
        ids1 = [seg.id for seg in segments]
        for grp in self.neuroml_morph._morphology.segment_groups:
            if grp.id == "apical_dendrites":
                #ids2 = [m.segments for m in grp.members]  # use this line once using Members with NeuroML
                ids2 = [m.id for m in grp.members]
                break
        self.assertEqual(ids1, ids2)

    def test_with_label_filter_neuromlmorph_section_name(self):
        filter = with_label("basal10")
        index = filter(self.neuroml_morph)
        assert index.size == 1
        segment = self.neuroml_morph.segments[index[0]]
        self.assertEqual(segment.name, "basal10")

    def test_with_label_filter_neuromlmorph_non_existent(self):
        filter = with_label("foo")
        self.assertRaises(ValueError, filter, self.neuroml_morph)


class NeuriteDistributionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not have_neuron:
            pytest.skip("NEURON not available")
        cls.array_morph = array_morph
        cls.neuroml_morph = neuroml_morph

    def test_uniform_neurite_distribution(self):
        value = 0.5
        absence = 0.0

        uniform_all = uniform('all', value, absence)
        for morph in self.array_morph, self.neuroml_morph:
            values  = np.array(
                [uniform_all.value_in(morph, i)
                 for i in range(len(morph))])
            self.assertEqual(values.sum(), value * len(morph))

        uniform_dend1 = uniform(apical_dendrites(), value, absence)
        values = np.array(
            [uniform_dend1.value_in(self.array_morph, i)
             for i in range(len(self.array_morph))])
        self.assertEqual(values.sum(), value * 8)  # 8 apical dendrite segments in array_morph

        uniform_dend2 = uniform(with_label("apical_dendrites"), value, absence)
        values = np.array(
            [uniform_dend2.value_in(self.neuroml_morph, i)
             for i in range(len(self.neuroml_morph))])
        self.assertEqual(values.sum(), value * 7)  # 7 apical dendrite segments in neuroml_morph

    def test_diameter_based_neurite_distribution(self):
        absence = 0.0

        triple_diam = by_diameter(with_label("basal_dendrites"),
                                  lambda d: 3 * d,
                                  absence=absence)
        self.assertEqual(triple_diam.value_in(self.neuroml_morph, 0),
                         absence)
        self.assertEqual(triple_diam.value_in(self.neuroml_morph, 1),
                         3 * self.neuroml_morph.get_diameter(1))

    def test___any__neurite_distribution(self):
        value_basal = 0.5
        value_apical = 0.6
        value_absence = 0.3

        uniform_basal = uniform(basal_dendrites(), value_basal)
        uniform_apical = uniform(apical_dendrites(), value_apical)

        distr = morph_any(uniform_basal, uniform_apical, absence=value_absence)
        self.assertEqual(distr.value_in(self.array_morph, 0),
                         value_absence) # soma
        self.assertEqual(distr.value_in(self.array_morph, 1),
                         value_basal)
        self.assertEqual(distr.value_in(self.array_morph, 12),
                         value_apical)
        self.assertEqual(distr.value_in(self.array_morph, 17),
                         value_absence)  # axon
