import re
import arbor
from pyNN.arborproto.procedures.swc_tags_names import get_swc_tag, extract_swc_tag_from_neuroml, get_name


def _create_base_dictionary(arbor_morph):
    base_dict = {"everywhere": "(all)", "root": "(root)", "terminal": "(terminal)"}
    for i in range(arbor_morph.num_branches):
        for j in range(len(arbor_morph.branch_segments(i))):
            tag = arbor_morph.branch_segments(i)[j].tag
            name = get_name(tag)
            base_dict.update({name: "(tag " + str(tag) + ")",
                              name + "_midpoint": "(restrict (on-branches 0.5) (tag " + str(tag) + "))"})
            if re.search("axon", name, re.IGNORECASE):
                # base_dict.update({name + "_terminal": "(restrict (locset 'terminal') (region '" + name + "'))"})
                base_dict.update({name + "_terminal": '(restrict (locset "terminal") (region "' + name + '"))'})
    return base_dict


def base_label(arbor_morph):
    return arbor.label_dict(_create_base_dictionary(arbor_morph))
