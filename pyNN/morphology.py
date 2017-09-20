"""



"""

import os.path
import shutil
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


def load_morphology(url):
    if os.path.exists(url):
        local_morph_file = url
    else:
        local_morph_file = _download_file(url)
    array_morph = neuroml.loaders.SWCLoader.load_swc_single(local_morph_file)
    return NeuroMLMorphology(array_morph)


class Morphology(object):
    """
    
    """

    def __init__(self):
        self.section_groups = {}

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

    @property
    def segments(self):
        return self._morphology.segments


class BrianMorphology(Morphology):
    """
    
    """
    pass



class IonChannelDistribution(object):
    pass


class uniform(IonChannelDistribution):

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



class MorphologyFilter(object):
    pass


class apical_dendrite(MorphologyFilter):

    def __init__(self, fraction_along=None):
        self.fraction_along = fraction_along

    def __call__(self, morphology):
        if APICALDENDRITE in morphology.section_groups:
            return morphology.section_groups[APICALDENDRITE]
        else:
            raise Exception("No neurites labelled as apical dendrite")


class random_section(MorphologyFilter):

    def __init__(self, f):
        self.f = f

    def __call__(self, morphology):
        sections = self.f(morphology)
        return numpy.random.choice(sections)