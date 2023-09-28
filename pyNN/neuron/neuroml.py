"""
Support cell types defined in NeuroML with NEURON.


:copyright: Copyright 2018 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from __future__ import absolute_import

import os
from collections import defaultdict
import neuroml
from neuroml.loaders import NeuroMLLoader
from neuron import h, nrn
from pyNN.models import BaseCellType
from pyNN.morphology import NeuroMLMorphology, with_label, by_distance
from .cells import NeuronTemplate, PROXIMAL, DISTAL
from .simulator import load_mechanisms
from pyNN.utility.build import compile_nmodl


class NeuroMLCell(NeuronTemplate):

    def read_morphology(self):
        # move this to the class factory, so that it is done only
        # once per cell type, not once per cell in the population
        assert isinstance(self.nml_cell.morphology, neuroml.Morphology)

        segments = {}
        segment_groups = {}
        cable_sections = []
        cable_section_members = {}

        # put segments into dictionary for easy look-up
        for seg_elem in self.nml_cell.morphology.segments:
            segments[seg_elem.id] = seg_elem

        # the following assumes we have segment groups marked with "sao864921383"
        # to denote cable sections
        for sg_elem in self.nml_cell.morphology.segment_groups:
            segment_groups[sg_elem.id] = {
                "members": [member.segments for member in sg_elem.members],
                "includes": [include.segment_groups for include in sg_elem.includes],
                "nodes": [],
                "parent": None
            }
            if sg_elem.neuro_lex_id == "sao864921383":
                # group defines a cable section
                cable_sections.append(sg_elem.id)
                for property in sg_elem.properties:
                    if property.tag == "numberInternalDivisions":
                        segment_groups[sg_elem.id]["nseg"] = int(property.value)
            if sg_elem.inhomogeneous_parameters:
                segment_groups[sg_elem.id]["inhomogeneous_parameters"] = {
                    param.id: param
                    for param in sg_elem.inhomogeneous_parameters
                }

        # check the assumption that we have either "members" (in cable section groups)
        # or "includes" (in other groups) but never both
        for segment_group_id, segment_group in segment_groups.items():
            if segment_group["members"]:
                assert not segment_group["includes"], "Not implemented"
                assert segment_group_id in cable_sections
            elif segment_group["includes"]:
                assert not segment_group["members"], "Not implemented"

        # build reverse lookup for membership of cable sections
        for cable_section_id in cable_sections:
            segment_group = segment_groups[cable_section_id]
            for segment_id in segment_group["members"]:
                cable_section_members[segment_id] = cable_section_id

        # process members
        for cable_section_id in cable_sections:
            segment_group = segment_groups[cable_section_id]
            for segment_id in segment_group["members"]:
                seg_elem = segments[segment_id]
                 # (note that pt3d accepts arrays in NEURON 7.5+ so we could make "nodes" an array)
                if seg_elem.proximal:
                    segment_group["nodes"].append(seg_elem.proximal)
                if seg_elem.distal:
                    segment_group["nodes"].append(seg_elem.distal)
                if seg_elem.parent is not None:  # root segment
                    if seg_elem.parent.segments not in segment_group["members"]:
                        # parent is in a different cable section
                        assert segment_group["parent"] is None
                        segment_group["parent"] = cable_section_members[seg_elem.parent.segments]
                        # todo: handle fraction_along

        # if we do not have segment groups marked with "sao864921383"
        # then treat each segment as a cable section
        if not cable_sections:
            raise NotImplementedError("todo")

        return segment_groups, cable_sections


    def __init__(self):
        # since we might wish to create a large population
        # should do NeuroML processing in the cell type
        # and produce an optimal data structure for creating Sections
        # efficiently (note that pt3d accepts arrays in NEURON 7.5+)

        self.traces = defaultdict(list)
        self.recording_time = False
        self.spike_source = None
        self.spike_times = h.Vector(0)
        self.v_init = None

        # create morphology
        self.sections = {}
        self.section_labels = {}

        segment_groups, cable_sections = self.read_morphology()

        self.root_section = None
        for cable_section_id in cable_sections:
            segment_group = segment_groups[cable_section_id]
            section = nrn.Section(name=cable_section_id)
            for node in segment_group["nodes"]:
                h.pt3dadd(node.x, node.y, node.z, node.diameter, sec=section)
            section.nseg = segment_group.get("nseg", 1)
            self.sections[cable_section_id] = section

            parent_id = segment_group["parent"]
            if parent_id is None:
                assert self.root_section is None
                self.root_section = cable_section_id
            else:
                connection_point = DISTAL
                # todo: handle "fraction_along" attribute rather than always using DISTAL
                section.connect(self.sections[parent_id], connection_point, PROXIMAL)

        # handle biophysics
        membrane_properties = self.nml_cell.biophysical_properties.membrane_properties
        self.set_section_property("cm", membrane_properties.specific_capacitances, segment_groups, "uF_per_cm2")

        # todo:
        # 'membrane_properties.channel_populations'

        for channel_density in membrane_properties.channel_densities:
            self.set_channel_properties(channel_density, segment_groups)

        for channel_density in membrane_properties.channel_density_ghk2s:
            raise NotImplementedError

        for channel_density in membrane_properties.channel_density_ghks:
            raise NotImplementedError

        for channel_density in membrane_properties.channel_density_nernsts:
            self.set_channel_properties(channel_density, segment_groups)

        for channel_density in membrane_properties.channel_density_non_uniform_ghks:
            raise NotImplementedError

        for channel_density in membrane_properties.channel_density_non_uniform_nernsts:
            raise NotImplementedError

        for channel_density in membrane_properties.channel_density_non_uniforms:
            self.set_channel_properties_non_uniform(channel_density, segment_groups)

        for channel_density in membrane_properties.channel_density_v_shifts:
            raise NotImplementedError

        if len(membrane_properties.init_memb_potentials) != 1:
            raise NotImplementedError
        if membrane_properties.init_memb_potentials[0].segment_groups != "all":
            raise NotImplementedError
        else:
            value, units = membrane_properties.init_memb_potentials[0].value.split()
            assert units == "mV"
            self.v_init = float(value)

        intracellular_properties = self.nml_cell.biophysical_properties.intracellular_properties
        # todo: handle intracellular_properties.species
        self.set_section_property("Ra", intracellular_properties.resistivities, segment_groups, "ohm_cm")

        # set source section
        if self.spike_source:
            self.source_section = self.sections[self.spike_source]
        elif "axon_initial_segment" in self.sections:
            self.source_section = self.sections["axon_initial_segment"]
        else:
            # take the root section
            self.source_section = self.sections[self.root_section]
        self.source = self.source_section(0.5)._ref_v
        self.rec = h.NetCon(self.source, None, sec=self.source_section)

    def set_section_property(self, property_name, element_list, segment_groups, expected_units):
        for elem in element_list:
            value, units = elem.value.split()
            if units != expected_units:
                raise ValueError("Unexpected units. Got {}, expected {}".format(units, expected_units))
            segment_group_id = elem.segment_groups
            if segment_group_id in self.sections:
                setattr(self.sections[segment_group_id], property_name, float(value))
                #print("Setting {}={} in {}".format(property_name, value, segment_group_id))
            else:
                for child_group_id in segment_groups[segment_group_id]["includes"]:
                    setattr(self.sections[child_group_id], property_name, float(value))
                    #print("Setting {}={} in {} (part of {})".format(property_name, value, child_group_id, segment_group_id))

    def set_channel_properties(self, element, segment_groups):
        segment_group_id = element.segment_groups
        gmax, units = element.cond_density.split()
        gmax = float(gmax)
        expected_units = "S_per_cm2"
        if units != expected_units:
             raise ValueError("Unexpected units. Got {}, expected {}".format(units, expected_units))
        if hasattr(element, "e_rev"):
            e_rev, units = element.e_rev.split()
            e_rev = float(e_rev)
            expected_units = "mV"
            if units != expected_units:
                raise ValueError("Unexpected units. Got {}, expected {}".format(units, expected_units))
            e_rev_name = {
                "na": "ena",
                "k": "ek",
                "ca": "eca",
                "hcn": "eh",
                "non_specific": "e"
            }[element.ion]
        else:
            e_rev_name = None

        mech_name = element.ion_channel
        if mech_name == "pas":  # special case, to avoid clashing with NEURON's built-in 'pas' mechanism
            mech_name += "_nml2"
        if segment_group_id in self.sections:
            self.sections[segment_group_id].insert(mech_name)
            for seg in self.sections[segment_group_id]:  # use allseg()?
                setattr(getattr(seg, mech_name), "gmax", gmax)
                if e_rev_name:
                    setattr(getattr(seg, mech_name), e_rev_name, e_rev)
        else:
            for child_group_id in segment_groups[segment_group_id]["includes"]:
                self.sections[child_group_id].insert(mech_name)
                for seg in self.sections[child_group_id]:  # use allseg()?
                    setattr(getattr(seg, mech_name), "gmax", gmax)
                    if e_rev_name:
                        setattr(getattr(seg, mech_name), e_rev_name, e_rev)

    def set_channel_properties_non_uniform(self, element, segment_groups):
        morph = NeuroMLMorphology(self.nml_cell.morphology)
        mech_name = element.ion_channel
        for elem in element.variable_parameters:
            ivalue = elem.inhomogeneous_value
            definition = segment_groups[elem.segment_groups]["inhomogeneous_parameters"][ivalue.inhomogeneous_parameters]
            expr = ivalue.value
            definition.variable  # for use in lambda expr
            if definition.metric != 'Path Length from root':
                raise NotImplementedError
            if definition.proximal:
                if definition.proximal.translation_start != 0.0:
                    raise NotImplementedError
            if definition.distal:
                if hasattr(definition.distal, "normalization_end"):
                    raise NotImplementedError
            # missing units? - ask Padraig
            distr = by_distance(with_label(elem.segment_groups),
                                eval("lambda {}: {}".format(definition.variable, expr)),
                                absence=None)

            param_name_map = {
                "condDensity": "gmax"
            }
            param_name = param_name_map[elem.parameter]
            for index, id in enumerate(self.sections):
                value = distr.value_in(morph, index)
                if value is not None:
                    self.sections[id].insert(mech_name)
                    for seg in self.sections[id]:
                        setattr(getattr(seg, mech_name), param_name, value)

        # todo: handle non-variable params, e.g. e_rev

        #<variableParameter segmentGroup="apical" parameter="condDensity">
        #    <inhomogeneousValue value="1e4 * ((-0.869600 + 2.087000*exp((p-0.000000)*0.003100))*0.000080)"
        #                        inhomogeneousParameter="PathLengthOver_apical"/>
        #</variableParameter>

    def memb_init(self):
        for state_var in ('v',):
            initial_value = getattr(self, '{0}_init'.format(state_var))
            assert initial_value is not None
            if state_var == 'v':
                for section in self.sections.values():
                    for seg in section:
                        seg.v = initial_value
            else:
                raise NotImplementedError()


class NeuroMLCellType(BaseCellType):
    units = {
        'v': 'mV',
    }

    def __init__(self, **parameters):
        BaseCellType.__init__(self, **parameters)

    @property
    def model(self):
        return type(self.nml_cell.id,
                    (NeuroMLCell,),
                    {"nml_cell": self.nml_cell})

    def can_record(self, variable, location=None):
        return True  # todo: implement this properly



def neuroml_cell_type(nml_cell):
    """
    Return a new NeuroMLCellType subclass.
    """
    name = nml_cell.id + "_CellType"
    return type(name, (NeuroMLCellType,), {"nml_cell": nml_cell})



def load_neuroml_cell_types(filename):
    doc = NeuroMLLoader.load(filename)
    # handle ion channels
    load_neuroml_ion_channels([include.href for include in doc.includes])
    # todo: ion channels defined directly in the main file, not included
    return [neuroml_cell_type(cell) for cell in doc.cells]


def load_neuroml_ion_channels(filenames):
    from pyneuroml import pynml
    from shutil import rmtree
    dirnames = set([])
    for filename in filenames:
        print("Handling {}".format(filename))
        pynml.run_lems_with_jneuroml_neuron(filename, nogui=True, only_generate_scripts=True,
                                            compile_mods = False, verbose=False)
        dirnames.add(os.path.abspath(os.path.dirname(filename)))
    for dirname in dirnames:
        if not os.path.exists(dirname + "/x86_64"):
            #rmtree(dirname + "/x86_64")  # tmp hack
            compile_nmodl(dirname)
            load_mechanisms(dirname)
