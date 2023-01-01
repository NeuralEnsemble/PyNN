# encoding: utf-8
"""
Functions to create Arbor labels.

:copyright: Copyright 2006-2022 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""
import re
import arbor
from pyNN.arbor.procedures.swc_tags_names import get_swc_tag, extract_swc_tag_from_neuroml


class _CreateArborLabels(object):
    """ This private class is called by CreateArborLabels (below) to invoke `_CreateArborLabels.create_labels`
    """
    def __init__(self, neuroml_morphology):
       self.neuromlmorph = neuroml_morphology

    def create_labels(self):
        """
        Call this as self.__create_arbor_labels() to create self._arbor_labels
        """
        return arbor.label_dict(self.__create_region_definitions(self.neuromlmorph))

    @staticmethod
    def __create_region_definitions(neuroml_morphology):
        # Arbor uses a domains specific language (DSL) to describe regions and locations, which are given labels.
        dict_defs = {"everywhere": "(all)", "root": "(root)", "terminal": "(terminal)"}
        for indx, nml_seg in enumerate(neuroml_morphology.backend_segments):
            tag_no = extract_swc_tag_from_neuroml(nml_seg.id, neuroml_morphology)  # tag_no = get_swc_tag(nml_seg)
            # print(nml_seg.name)
            dict_defs.update({nml_seg.name: "(tag " + str(tag_no) + ")"})
            # (restrict (on-branches 0.5) (tag 3)) for spike detection and voltage probe for undefined locations
            dict_defs.update({nml_seg.name + "_midpoint": "(restrict (on-branches 0.5) (tag " + str(tag_no) + "))"})
            if re.search("axon", nml_seg.name, re.IGNORECASE):
                # "axon_terminal": '(restrict (locset "terminal") (region "axon"))'
                dict_defs.update({nml_seg.name + "_terminal":
                                      "(restrict (locset 'terminal') (region '" + nml_seg.name + "'))"})
        return dict_defs


class CreateArborLabels(object):
    """
    Use:
    ```
    from pyNN.arbor.procedures.step4 import CreateArborLabels
    self._arbor_labels = CreateArborLabels.create_labels(self.morphology)
    ```
    """
    def __new__(cls, neuroml_morphology):
        cal = _CreateArborLabels(neuroml_morphology)
        return cal.create_labels()