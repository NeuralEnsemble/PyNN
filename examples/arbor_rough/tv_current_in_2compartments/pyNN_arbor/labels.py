# ~/mc/pyNN/arbor/labels.py

import re
import arbor

def get_tag(nml_seg):
    """
    SWC based specification.
    http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    """
    if re.search("soma", nml_seg.name, re.IGNORECASE):
        return 1
    elif re.search("axon", nml_seg.name, re.IGNORECASE):
        return 2
    elif re.search("dend", nml_seg.name, re.IGNORECASE):
        return 3
    else:
        return 5
        
def create_dict(pynn_nml_morph):
    """
    Arbor uses a domains specific language (DSL) to describe regions and locations, which are given labels.
    https://arbor.readthedocs.io/en/latest/concepts/labels.html?
    """
    dict_defs = {}
    for i, nml_seg in enumerate(pynn_nml_morph.segments):
        dict_defs.update({nml_seg.name: "(tag "+ str(get_swc_tag(nml_seg))+ ")"})
        dict_defs.update({"everywhere": "(all)"})
    return dict_defs

