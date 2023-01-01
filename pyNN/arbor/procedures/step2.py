# encoding: utf-8
"""
Functions to set/configure morphology.

:copyright: Copyright 2006-2022 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""
import neuroml
import numpy
from pyNN.arbor.procedures.swc_tags_names import get_swc_tag


class ConfigureMorphology(object):
    """
    Use:
    ```
    from pyNN.arbor.procedures.step2 import ConfigureMorphology
    self.morphology = ConfigureMorphology.set_soma_index(self.morphology)
    ```
    Furthermore, when `isinstance(self.morphology._morphology, neuroml.Morphology)`
    ```
    self.morphology = ConfigureMorphology.include_section_groups(self.morphology)
    ```
    """
    @staticmethod
    def set_soma_index(neuroml_morphology):
        if isinstance(neuroml_morphology._morphology, neuroml.arraymorph.ArrayMorphology):
            neuroml_morphology._soma_index = 0  # fragile temporary hack - should be index of the vertex with no parent
        elif isinstance(neuroml_morphology._morphology, neuroml.Morphology):
            for index, segment in enumerate(neuroml_morphology.backend_segments):
                if segment.name == "soma":
                    neuroml_morphology._soma_index = segment.id
        return neuroml_morphology

    @staticmethod
    def include_section_groups(neuroml_morphology):
        # For use when NOT isinstance(self.morphology._morphology, neuroml.arraymorph.ArrayMorphology)
        section_groups = {}
        for indx, nml_seg in enumerate(neuroml_morphology.backend_segments):
            tag = get_swc_tag(nml_seg)
            if tag not in section_groups.keys():
                section_groups.update({tag: numpy.array(indx)})
            else:
                section_groups[tag] = numpy.append(section_groups[tag], indx)
        neuroml_morphology.section_groups = section_groups
        return neuroml_morphology
