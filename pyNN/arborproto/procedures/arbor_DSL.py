# encoding: utf-8
"""
Functions to get Arbor DSL form.

:copyright: Copyright 2006-2022 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN.arborproto.procedures.swc_tags_names import get_swc_tag


def get_region_DSL(arg):
    if arg == "nil":
        return "(region-nil)"
    elif arg == "all":
        return "(all)"
    elif arg != "nil" and arg != "all" and arg is str:
        return "(tag " + str(get_swc_tag(arg)) + ")"
    # arbor branch is not used in PyNN (branch tag_int)
    # arbor segment is not used in PyNN (segment tag_int)
    # For more see: https://docs.arbor-sim.org/en/latest/concepts/labels.html?highlight=dsl#region-expressions


def get_locset_DSL(synparameters, arbormorph):
    density_func_name = synparameters["density"].__class__.__name__
    loc_selector = synparameters["density"].selector
    if density_func_name == "uniform":
        um = synparameters["density"].value
        return get_synaptic_locset_DSL_uniformly_per_um(loc_selector, um, arbormorph)
    # For more see: https://docs.arbor-sim.org/en/latest/concepts/labels.html?highlight=dsl#locset-expressions


def get_synaptic_locset_DSL_uniformly_per_um(loc_selector, um, arbormorph):
    var_um = um  # 0.01 corresponds to 1 μm
    ans = ""
    if loc_selector == "all":
        for branch in range(arbormorph.num_branches):
            while var_um <= 1:
                ans = ans + "(location " + str(branch) + " " + str(var_um) + ") "
                var_um = var_um + um  # update
            var_um = um  # reset
        return "(join " + ans + ")"
    else:
        tag_id = get_swc_tag(loc_selector)
        while var_um <= 1:
            ans = ans + "(restrict (on-branches " + str(var_um) + ") (tag " + str(tag_id) + "))"
            var_um = var_um + um  # update
        return "(join " + ans + ")"  # (restrict (on-branches 0.5) (tag 3))


def get_synaptic_locset_DSL_distance_per_um(density_func):
    #  by_distance(dendrites(), lambda d: 0.05 * (d < 50.0)),  # number per µm
    loc_selector = density_func.selector.__class__.__name__
    tag_id = get_swc_tag(loc_selector)
    um = 0.01  # initialization; 0.01 corresponds to 1 μm
    var_um = um
    ans = ""
    while var_um <= 1:
        dist = density_func.distance_function(var_um)
        ans = ans + "(restrict (on-branches " + str(dist) + ") (tag " + str(tag_id) + "))"
        var_um = var_um + um  # update
    return "(join " + ans + ")"
