"""



"""

import os.path
import shutil
import numpy as np
import numpy.random
import neuroml.loaders

# swc compartment types
UNDEFINED = 0
SOMA = 1
AXON = 2
BASALDENDRITE = 3
APICALDENDRITE = 4
CUSTOM = 5
UNSPECIFIEDNEURITES = 6
GLIAPROCESSES = 7


def _download_file(url):
    import requests  # consider rewriting using just standard library
                     # to avoid adding another dependency
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return local_filename


def load_morphology(url, replace_axon=False):
    if os.path.exists(url):
        local_morph_file = url
    else:
        local_morph_file = _download_file(url)
    # todo: handle replace_axon argument
    array_morph = neuroml.loaders.SWCLoader.load_swc_single(local_morph_file)
    return NeuroMLMorphology(array_morph)


class Morphology(object):
    """
    
    """

    def __init__(self):
        self.section_groups = {}
        self.synaptic_receptors = {}

    @property
    def soma_index(self):
        return self._soma_index



class NeuroMLMorphology(Morphology):
    """
    
    """

    def __init__(self, morphology):
        super(NeuroMLMorphology, self).__init__()
        self._morphology = morphology
        if isinstance(morphology, neuroml.arraymorph.ArrayMorphology):
            for neurite_type in (AXON, BASALDENDRITE, APICALDENDRITE):
                self.section_groups[neurite_type] = (morphology.node_types == neurite_type).nonzero()[0]
        self._path_lengths = None

    @property
    def segments(self):
        return self._morphology.segments

    @property
    def path_lengths(self):
        if self._path_lengths is None:
            if isinstance(self._morphology, neuroml.arraymorph.ArrayMorphology):
                M = self._morphology
                parent_vertices = M.vertices[:-1][M.connectivity[1:]]
                lengths = M.vertices[1:, 0] - parent_vertices[:, 0]  # just x-distance for now, need to fix
                path_lengths = np.zeros_like(M.connectivity)
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



class IonChannelDistribution(object):
    pass


class SynapseDistribution(object):
    pass


class uniform(IonChannelDistribution, SynapseDistribution):
    # we inherit from two parents, because we want to use the name "uniform" for both
    # the implementation behaves differently depending on context
    # we could perhaps just have a single parent, e.g. NeuriteDistribution

    def __init__(self, selector, value):
        self.selector = selector
        self.value = value

    def value_in(self, morphology, index):
        if self.selector == 'all':
            return self.value
        elif self.selector == 'soma':
            if index == morphology.soma_index:
                return self.value
            else:
                return 0.0
        else:
            raise NotImplementedError("selector '{}' not yet supported".format(self.selector))


class by_distance(IonChannelDistribution, SynapseDistribution):

    def __init__(self, selector, distance_function):
        self.selector = selector
        self.distance_function = distance_function

    def value_in(self, morphology, index):
        if isinstance(self.selector, MorphologyFilter):
            selected_indices = self.selector(morphology)  # need to cache this, or allow index to be an array
            if index in selected_indices:
                distance = morphology.get_distance(index)
                return self.distance_function(distance)
            else:
                return 0.0
        else:
            raise NotImplementedError("selector '{}' not yet supported".format(self.selector))


class MorphologyFilter(object):
    pass



class dendrites(MorphologyFilter):

    def __init__(self, fraction_along=None):
        self.fraction_along = fraction_along

    def __call__(self, morphology, filter_by_receptor_type=False):
        ids = np.array([], dtype=int)
        for label in (APICALDENDRITE, BASALDENDRITE):
            if label in morphology.section_groups:
                ids = np.hstack((ids, morphology.section_groups[label]))
        if ids.size < 1:
            raise Exception("No neurites labelled as dendrites")
        return ids


class apical_dendrites(MorphologyFilter):

    def __init__(self, fraction_along=None):
        self.fraction_along = fraction_along

    def __call__(self, morphology, filter_by_receptor_type=False):
        # if filter_by_receptor_type is not False,
        # return only sections that contain at least one post-synaptic receptor
        # of the specified name
        if APICALDENDRITE in morphology.section_groups:
            sections = morphology.section_groups[APICALDENDRITE]
            if filter_by_receptor_type:
                sections = np.intersect1d(sections,
                                          np.fromiter(morphology.synaptic_receptors[filter_by_receptor_type].keys(), dtype=int))
            return sections
        else:
            raise Exception("No neurites labelled as apical dendrite")


class random_section(MorphologyFilter):

    def __init__(self, f):
        self.f = f

    def __call__(self, morphology, **kwargs):
        sections = self.f(morphology, **kwargs)
        return numpy.random.choice(sections)
