# encoding: utf-8
"""
Functions to create Arbor morphology.

:copyright: Copyright 2006-2022 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""
import arbor
from pyNN.arbor.procedures.swc_tags_names import extract_swc_tag_from_neuroml


class _CreateArborMorphology(object):
    """This private class is called by CreateArborMorphology (below) to invoke `_CreateArborMorphology.create_morphology`
    """
    def __init__(self, neuroml_morphology):
        self.neuromlmorph = neuroml_morphology

    def create_morphology(self):
        arbor_tree = self.__create_arbor_tree()
        return arbor_tree, arbor.morphology(arbor_tree)

    def __create_arbor_tree(self):
        # Creates tree directly from file if available.
        tree = arbor.segment_tree()
        for i, nml_seg in enumerate(self.neuromlmorph.backend_segments):
            self.__append_arbor_tree(tree, nml_seg)
        self._arbor_tree = tree  # arbor tree segment ~ NEURON section i.e unbranched collection of segments
        return tree

    def __append_arbor_tree(self, tree, nml_seg):
        if not nml_seg.parent:
            tree.append(arbor.mnpos,
                        arbor.mpoint(nml_seg.proximal.x, nml_seg.proximal.y, nml_seg.proximal.z,
                                     nml_seg.proximal.diameter / 2),
                        arbor.mpoint(nml_seg.distal.x, nml_seg.distal.y, nml_seg.distal.z,
                                     nml_seg.distal.diameter / 2),
                        tag=extract_swc_tag_from_neuroml(nml_seg.id, self.neuromlmorph))
        else:
            tree.append(nml_seg.parent.id,
                        arbor.mpoint(nml_seg.proximal.x, nml_seg.proximal.y, nml_seg.proximal.z,
                                     nml_seg.proximal.diameter / 2),
                        arbor.mpoint(nml_seg.distal.x, nml_seg.distal.y, nml_seg.distal.z,
                                     nml_seg.distal.diameter / 2),
                        tag=extract_swc_tag_from_neuroml(nml_seg.id, self.neuromlmorph))


class CreateArborMorphology(object):
    """
    Use:
    ```
    from pyNN.arbor.procedures.step3 import CreateArborMorphology
    self._arbor_morphology = CreateArborMorphology.create_morphology(self.morphology)
    ```
    """

    def __new__(cls, neuroml_morphology):
        cam = _CreateArborMorphology(neuroml_morphology)
        return cam.create_morphology()
