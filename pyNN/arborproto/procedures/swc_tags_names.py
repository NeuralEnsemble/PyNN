# encoding: utf-8
"""
Functions to get swc Tags (integer) and their corresponding names (string).

:copyright: Copyright 2006-2022 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""
import neuroml
import re
import pyNN.morphology as pyNNmorph


def get_swc_tag(nml_seg_or_string_name):
    if isinstance(nml_seg_or_string_name, neuroml.nml.nml.Segment):
        name = nml_seg_or_string_name.name
    else:
        name = nml_seg_or_string_name
    if re.search("soma", name, re.IGNORECASE):
        return pyNNmorph.SOMA  # = 1
    elif re.search("axon", name, re.IGNORECASE):
        return pyNNmorph.AXON  # = 2
    elif (re.search("basal", name, re.IGNORECASE)
          and re.search("dend", name, re.IGNORECASE)):
        return pyNNmorph.BASALDENDRITE  # = 3
    elif (re.search("apical", name, re.IGNORECASE)
          and re.search("dend", name, re.IGNORECASE)):
        return pyNNmorph.APICALDENDRITE  # = 4
    elif re.search("custom", name, re.IGNORECASE):
        return pyNNmorph.CUSTOM  # = 5
    elif re.search("neurite", name, re.IGNORECASE):
        return pyNNmorph.UNSPECIFIEDNEURITES  # = 6
    elif re.search("glia", name, re.IGNORECASE):
        return pyNNmorph.GLIAPROCESSES  # = 7
    else:
        return pyNNmorph.UNDEFINED  # = 0


def get_name(swc_tag):
    tag_and_name = {
        0: "UNDEFINED",
        1: "soma",
        2: "axon",
        3: "basal_dendrite",
        4: "apical_dendrite",
        5: "CUSTOM",
        6: "unspecific_neurite",
        7: "glia",
    }
    return tag_and_name[swc_tag]


def extract_swc_tag_from_neuroml(seg_id_as_array_element, neuroml_morph):
    ans_swc_tag = None
    for swc_tag in neuroml_morph.section_groups.keys():
        if seg_id_as_array_element in neuroml_morph.section_groups[swc_tag]:
            ans_swc_tag = swc_tag
    if ans_swc_tag is None:
        return seg_id_as_array_element  # This in general is 1
    else:
        return ans_swc_tag

def extract_swc_tag_from_neuroml2(distal_point, neuroml_morph):
    pass