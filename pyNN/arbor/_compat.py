"""
Compatibility shims for supporting a range of Arbor versions from a single codebase.

:copyright: Copyright 2006-2026 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import arbor
from arbor import units as U


# --- iclamp was renamed to i_clamp in Arbor 0.12 -----------------------------

_MECHANISM_RENAMES = {"iclamp": "i_clamp"}


def get_electrode_mechanism(name):
    """Return the Arbor current-source mechanism class for the given PyNN model
    name, accounting for renames across Arbor versions (e.g. iclamp -> i_clamp)."""
    for candidate in (name, _MECHANISM_RENAMES.get(name)):
        if candidate is not None and hasattr(arbor, candidate):
            return getattr(arbor, candidate)
    raise AttributeError(f"Arbor has no current-source mechanism {name!r}")


# --- decor.place() dropped the label arg for current stimuli in Arbor 0.12 ----

def _current_stimulus_place_takes_label():
    decor = arbor.decor()
    clamp = get_electrode_mechanism("iclamp")(0.0 * U.ms, 0.0 * U.ms, 0.0 * U.nA)
    try:
        decor.place("(root)", clamp, "_probe")
        return True
    except TypeError:
        return False


_CURRENT_PLACE_TAKES_LABEL = _current_stimulus_place_takes_label()


def place_current_source(decor, locset, mechanism, label):
    """Place a current-clamp stimulus on a decor. Arbor 0.12 dropped the label
    argument from the current-stimulus ``place()`` overload that 0.10 accepted
    (synapse/junction/detector placements keep their label in all versions)."""
    if _CURRENT_PLACE_TAKES_LABEL:
        decor.place(locset, mechanism, label)
    else:
        decor.place(locset, mechanism)


# --- cv_policy_max_extent gained units in Arbor 0.12 -------------------------

def _cv_policy_wants_units():
    try:
        arbor.cv_policy_max_extent(1.0 * U.um)
        return True
    except TypeError:
        return False


_CV_POLICY_WANTS_UNITS = _cv_policy_wants_units()


def max_extent_policy(length_um):
    """Build a max-extent cv_policy from a length in micrometres, handling the
    0.12 change that requires a unit-typed quantity."""
    if _CV_POLICY_WANTS_UNITS:
        return arbor.cv_policy_max_extent(length_um * U.um)
    return arbor.cv_policy_max_extent(length_um)


# --- discretisation moved from decor to cable_cell in Arbor 0.11 -------------

_DECOR_HAS_DISCRETIZATION = hasattr(arbor.decor(), "discretization")


def make_cable_cell(tree, decor, labels, discretization):
    """Construct an ``arbor.cable_cell``, applying the discretisation cv_policy in
    the way the installed Arbor version expects: as a ``decor`` method in 0.10, or
    as a ``cable_cell`` constructor argument in 0.11+."""
    if _DECOR_HAS_DISCRETIZATION:
        decor.discretization(discretization)
        return arbor.cable_cell(tree, decor, labels)
    return arbor.cable_cell(tree, decor, labels, discretization=discretization)
