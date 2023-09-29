"""



"""

from __future__ import absolute_import
import os.path
import shutil
import numpy as np
try:
    import neuroml.arraymorph
    import neuroml.loaders
    have_neuroml = True
except ImportError:
    have_neuroml = False
import morphio
from morphio import SectionType


def _download_file(url):
    import requests  # consider rewriting using just standard library
                     # to avoid adding another dependency
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return local_filename


def load_morphology(url, replace_axon=False, use_library="neuroml"):
    if os.path.exists(url):
        local_morph_file = url
    else:
        local_morph_file = _download_file(url)
    # todo: handle replace_axon argument
    # todo: fix load_swc to handle "standardized" somas as described
    #       in http://neurom.readthedocs.io/en/latest/definitions.html#soma
    #       and http://www.neuromorpho.org/SomaFormat.html
    if use_library == "neuroml":
        if have_neuroml:
            array_morph = neuroml.loaders.SWCLoader.load_swc_single(local_morph_file)
        else:
            raise ImportError("Please install libNeuroML")
        return NeuroMLMorphology(array_morph)
    elif use_library == "morphio":
        return MorphIOMorphology(local_morph_file)


class Morphology(object):
    """

    """
    is_lazyarray_scalar = True

    def __init__(self):
        self.section_groups = {}

    @property
    def soma_index(self):
        return self._soma_index



class NeuroMLMorphology(Morphology):
    """

    """

    def __init__(self, morphology):
        if not have_neuroml:
            raise ImportError("Please install libNeuroML to use the NeuroMLMorphology class")
        super(NeuroMLMorphology, self).__init__()
        self._morphology = morphology
        if isinstance(morphology, neuroml.arraymorph.ArrayMorphology):
            for neurite_type in (SectionType.soma, SectionType.axon, SectionType.basal_dendrite, SectionType.apical_dendrite):
                self.section_groups[neurite_type] = (morphology.node_types == neurite_type).nonzero()[0]
        elif isinstance(morphology, neuroml.Morphology):
            self.id_map = {seg.id: i
                           for i, seg in enumerate(morphology.segments)}
            for grp in morphology.segment_groups:
                self.section_groups[grp.id] = np.array([self.id_map[seg.id] for seg in grp.members])
        self._path_lengths = None

    def __len__(self):
        if isinstance(self._morphology, neuroml.arraymorph.ArrayMorphology):
            return self._morphology.vertices.shape[0] - 1
        else:
            return len(self._morphology.segments)

    @property
    def segments(self):
        return self._morphology.segments

    def labels(self):
        _labels = {}
        for i, segment in enumerate(self.segments):
            _labels[segment.name] = i
        return _labels

    @property
    def soma_index(self):
        return self.labels().get("soma", 0)  # todo: more robust way to handle morphologies without a declared soma, e.g. single dendrites

    @property
    def path_lengths(self):
        if self._path_lengths is None:
            if isinstance(self._morphology, neuroml.arraymorph.ArrayMorphology):
                M = self._morphology
                parent_vertices = M.vertices[:-1][M.connectivity[1:]]
                lengths = np.zeros(((parent_vertices.shape[0],)), dtype=float)
                for axis in (0, 1, 2):
                    lengths += (M.vertices[1:, axis] - parent_vertices[:, axis])**2
                lengths = np.sqrt(lengths)
                path_lengths = np.zeros_like(M.connectivity, dtype=float)
                # can probably vectorize the following
                for i in range(1, path_lengths.size):
                    path_lengths[i] = path_lengths[M.connectivity[i]] + lengths[i - 1]
            else:
                raise NotImplementedError()
            self._path_lengths = path_lengths
        return self._path_lengths

    def get_distance(self, index):
        # for now, distance is to the proximal vertex.
        # could add option to return distance to centre of section or distal vertex
        if isinstance(self._morphology, neuroml.arraymorph.ArrayMorphology):
            return self.path_lengths[index]
        else:
            raise NotImplementedError()

    def get_diameter(self, index, fraction_along=0.0):
        if isinstance(self._morphology, neuroml.arraymorph.ArrayMorphology):
            raise NotImplementedError()
        else:
            return self.segments[index].proximal.diameter  # for now, diameter of proximal vertex


# todo: turn the following implementation of path_lengths into a unit test
# vertices = np.array([0.0, 2.0, 4.1, 6.3, 8.6, 6.4, 8.8])
# connectivity = np.array([-1, 0, 1, 2, 3, 2, 5])
# parent_vertices = vertices[:-1][connectivity[1:]]
# lengths = vertices[1:] - parent_vertices
# path_lengths = np.zeros_like(connectivity)
# for i in range(1, path_lengths.size):
#     path_lengths[i] = path_lengths[connectivity[i]] + lengths[i - 1]
# np.testing.assert_array_equal(path_lengths, np.array([0.0, 2.0, 4.1, 6.3, 8.6, 6.4, 8.8]))


class BrianMorphology(Morphology):
    """

    """
    pass


class MorphIOMorphology(Morphology):

    def __init__(self, morphology_file):
        super().__init__()
        self.morphology_file = morphology_file
        self._morphology = morphio.Morphology(morphology_file)
        for neurite_type in (SectionType.axon, SectionType.basal_dendrite, SectionType.apical_dendrite):
            self.section_groups[neurite_type] = (self._morphology.section_types == neurite_type).nonzero()[0]


class NeuriteDistribution(object):

    def __init__(self, selector, value_provider, absence=0.0):
        if isinstance(selector, MorphologyFilter):
            self.selector = selector
        elif isinstance(selector, str):
            self.selector = self.get_with_label_selector(selector)
        else:
            raise TypeError("'selector' should be either a MorphologyFilter or a string")
        self.value_provider = value_provider
        self.absence = absence


class IonChannelDistribution(NeuriteDistribution):
    pass


class SynapseDistribution(NeuriteDistribution):
    pass



class uniform(IonChannelDistribution, SynapseDistribution):
    # we inherit from two parents, because we want to use the name "uniform" for both
    # the implementation behaves differently depending on context
    # we could perhaps just have a single parent, e.g. NeuriteDistribution
    pass


class by_distance(IonChannelDistribution, SynapseDistribution):
    pass


class by_diameter(IonChannelDistribution, SynapseDistribution):
    """Distribution as a function of neurite diameter."""
    pass


class any(IonChannelDistribution, SynapseDistribution):
    """From a list of NeuriteDistribution objects,
    return the value from the first that matches the selector.
    """

    def __init__(self, *distributions, **kwargs):
        self.distributions = distributions
        self.absence = kwargs.get("absence", None)

    def value_in(self, morphology, index):
        for distribution in self.distributions:
            value = distribution.value_in(morphology, index)
            if value != distribution.absence:
                return value
        return self.absence



class MorphologyFilter(object):
    pass



class dendrites(MorphologyFilter):

    def __init__(self, fraction_along=None):
        self.fraction_along = fraction_along


class apical_dendrites(MorphologyFilter):

    def __init__(self, fraction_along=None):
        self.fraction_along = fraction_along


class basal_dendrites(MorphologyFilter):

    def __init__(self, fraction_along=None):
        self.fraction_along = fraction_along


class axon(MorphologyFilter):

    def __init__(self, fraction_along=None):
        self.fraction_along = fraction_along


class random_section(MorphologyFilter):

    def __init__(self, f):
        self.f = f


sample = random_section  # alias


class with_label(MorphologyFilter):
    """
    Select sections by label.

    Values will be matched against section group names
    then against individual section names.

    Example:

    with_label("soma", "dend", "axon")
    """

    def __init__(self, *labels):
        self.labels = labels



class LocationGenerator:

    def lazily_evaluate(self, mask=None, shape=None):
        return self


class LabelledLocations(LocationGenerator):

    def __init__(self, *labels):
        for label in labels:
            assert isinstance(label, str)
        self.labels = labels


class at_distances(LocationGenerator):
    # fractional distances, 0-1

    def __init__(self, selector, distances):
        if isinstance(selector, MorphologyFilter):
            self.selector = selector
        elif isinstance(selector, str):
            self.selector = self.get_with_label_selector(selector)
        else:
            raise TypeError("'selector' should be either a MorphologyFilter or a string")
        self.distances = distances


class random_placement(LocationGenerator):

    def __init__(self, density_function):
        self.density_function = density_function


class centre(LocationGenerator):

    def __init__(self, selector):
        if isinstance(selector, MorphologyFilter):
            self.selector = selector
        elif isinstance(selector, str):
            self.selector = self.get_with_label_selector(selector)
        else:
            raise TypeError("'selector' should be either a MorphologyFilter or a string")
