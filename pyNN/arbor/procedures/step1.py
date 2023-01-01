# encoding: utf-8
"""
Functions to create backend_segments attribute to morphology object.

:copyright: Copyright 2006-2022 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""
import numpy
from pyNN.arbor.procedures.swc_tags_names import get_name, extract_swc_tag_from_neuroml


class _CreateBackendSegments(object):  # SHOULD THIS BE IN morphology.py API?
    """ This private class is called by CreateBackendSegments (below) to invoke
    `_CreateBackendSegments.adjust_segments and _CreateBackendSegments.add_segment_names`
    """

    def __init__(self, neuroml_morphology):
        self.neuromlmorph = neuroml_morphology

    #########################################################
    # Creates attribute backend_segments with adjusted values
    #########################################################
    def adjust_segments(self):  # SHOULD THIS ALSO BE IN morphology.py API?
        # cls.morphology = neuroml_morphology
        prox, dist, parn = self.__get_segments_info()
        print(prox, dist, parn)
        setattr(self.neuromlmorph, "backend_segments", self.__init_backend_segments(len(parn), self.neuromlmorph))
        for i in range(len(parn)):
            setattr(self.neuromlmorph.backend_segments[i].proximal, "x", prox[i][0])
            setattr(self.neuromlmorph.backend_segments[i].proximal, "y", prox[i][1])
            setattr(self.neuromlmorph.backend_segments[i].proximal, "z", prox[i][2])
            setattr(self.neuromlmorph.backend_segments[i].proximal, "diameter", prox[i][3])
            setattr(self.neuromlmorph.backend_segments[i].distal, "x", dist[i][0])
            setattr(self.neuromlmorph.backend_segments[i].distal, "y", dist[i][1])
            setattr(self.neuromlmorph.backend_segments[i].distal, "z", dist[i][2])
            setattr(self.neuromlmorph.backend_segments[i].distal, "diameter", dist[i][3])
            # Unlike for segments[i].id in backend_segments[i].id corresponds to swctag because backend_segments will
            # correspond to segments in arbor tree
            # setattr(self.neuromlmorph.backend_segments[i], "id", extract_swc_tag_from_neuroml)
            #
            if parn[i] is None:
                if not not self.neuromlmorph.backend_segments[i].parent:
                    setattr(self.neuromlmorph.backend_segments[i], "parent", None)
            else:
                setattr(self.neuromlmorph.backend_segments[i].parent, "id", parn[i])
        return self.neuromlmorph

    @staticmethod
    def __init_backend_segments(n_new, neuroml_morphology):
        back_seg = list(range(n_new))
        for i in range(n_new):
            back_seg[i] = neuroml_morphology.segments[i]
        return back_seg

    def __get_segments_info(self):  # SHOULD THIS ALSO BE IN morphology.py API?
        arraymorph = self.neuromlmorph._morphology  # arraymorph = cls.morphology._morphology
        # setup
        prox = arraymorph.vertices[0]  # initialize array of proximal points of each segment
        dist = arraymorph.vertices[1]  # initialize array of distal points of each segment
        parn_dist = arraymorph.vertices[1][:3]  # initialize array of only distal 3D points for each segment
        parn = [None]  # initialize array of parent id for each segment
        for i in range(2, len(arraymorph.vertices) - 1):
            if arraymorph.connectivity[i] < arraymorph.connectivity[i + 1]:
                prox = numpy.vstack((prox, arraymorph.vertices[i]))  # creates column vector of m = no. of segments
                dist = numpy.vstack((dist, arraymorph.vertices[i + 1]))  # ditto
                parn = self.__add_parent_id(parn, parn_dist, arraymorph.vertices[i][:3], prox[0, :3])  # appends list
                parn_dist = numpy.vstack((parn_dist, arraymorph.vertices[i + 1][:3]))  # column vector of shape (m,)
            elif arraymorph.connectivity[i] == arraymorph.connectivity[i + 1]:
                j = i - 1
                while arraymorph.connectivity[j] == arraymorph.connectivity[i + 1]:
                    j = j - 1
                prox = numpy.vstack((prox, arraymorph.vertices[j]))
                dist = numpy.vstack((dist, arraymorph.vertices[i + 1]))
                parn = self.__add_parent_id(parn, parn_dist, arraymorph.vertices[j][:3], prox[0, :3])
                parn_dist = numpy.vstack((parn_dist, arraymorph.vertices[i + 1][:3]))
            elif arraymorph.connectivity[i + 1] == 0:
                j = arraymorph.connectivity[i + 1]
                prox = numpy.vstack((prox, arraymorph.vertices[j]))
                dist = numpy.vstack((dist, arraymorph.vertices[i + 1]))
                parn = self.__add_parent_id(parn, parn_dist, arraymorph.vertices[j][:3], prox[0, :3])
                parn_dist = numpy.vstack((parn_dist, arraymorph.vertices[i + 1][:3]))
        return prox, dist, parn  # all three arrays must have same length

    @staticmethod  # SHOULD THIS ALSO BE IN morphology.py API?
    def __add_parent_id(parn, parn_dist, points_3d, points_root):
        # M3.vertices[0][:3].shape[0] == M3.vertices[0][:3].size
        if parn_dist.shape[0] == parn_dist.size:
            if numpy.count_nonzero(parn_dist == points_3d) == 3:
                k = 0
                parn.append(k)
        else:
            # chk = (parn_dist == M3.vertices[i+1][:3]).all(axis=1)
            chk = (parn_dist == points_3d).all(axis=1)
            if numpy.count_nonzero(chk) != 0:
                k = numpy.where((parn_dist == points_3d).all(1))[0][0]
                parn.append(k)
            elif numpy.count_nonzero(chk) == 0:
                if numpy.count_nonzero(points_3d == points_root) == 3:
                    parn.append(None)
        return parn

    ######################################################################
    # Add (segment) names to the above created morphology.backend_segments
    ######################################################################
    @staticmethod
    def add_segment_names(backsegmented_neuroml_morphology):  # SHOULD THIS ALSO BE IN morphology.py API?
        # For use when isinstance(self.morphology._morphology, neuroml.arraymorph.ArrayMorphology)
        for indx, nml_seg in enumerate(backsegmented_neuroml_morphology.backend_segments):
            for swc_tag, array_morph in backsegmented_neuroml_morphology.section_groups.items():
                if nml_seg.id in array_morph:
                    setattr(nml_seg, "name", get_name(swc_tag))  # nml_seg.name = self.__get_name(swc_tag)
            if nml_seg.name is None:  # if it is still None, then it is soma segment i.e. swc tag = 1
                setattr(nml_seg, "name", get_name(1))  # nml_seg.name = self.__get_name(1)
        return backsegmented_neuroml_morphology


class CreateBackendSegments(object):
    """
    This class is called when `isinstance(self.morphology._morphology, neuroml.arraymorph.ArrayMorphology)`
    Use:
    ```
    from pyNN.arbor.procedures.step1 import CreateBackendSegments
    self.morphology = CreateBackendSegments(self.morphology)
    ```
    This returns the same morphology object but with the addition of attribute `backend_segments`.
    """

    def __new__(cls, neuroml_morphology):
        cbs = _CreateBackendSegments(neuroml_morphology)
        new_neuroml_morphology = cbs.adjust_segments()
        return cbs.add_segment_names(new_neuroml_morphology)
