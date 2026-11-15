# todo: I'm not sure it's a good idea to have a separate morphology module for each backend
# the scripts would look nicer if we had just the top-level morphology module and
# then used the function arguments to provide simulator-specificity

import arbor
from morphio import SectionType
from .. import morphology as base_morphology


# --- MorphologyFilters ---

def _region_expression(selector, morphology):
    """Return a region expression for `selector` that does not refer to labels.

    Label references are not resolved in probe locsets (and can crash Arbor 0.12),
    so where possible a label is replaced by the segments it refers to.
    """
    region = selector.get_region(morphology)
    if region.startswith('"'):
        return f"(region {region})"
    return region


class with_label(base_morphology.with_label):

    def get_region(self, morphology=None):
        if len(self.labels) > 1:
            raise NotImplementedError
        assert len(self.labels) != 0
        label = self.labels[0]
        segments = getattr(morphology, "segments", None)  # NeuroML morphologies only
        if segments:
            # segment indices as in CellDescriptionBuilder._build_tree()
            indices = [i for i, segment in enumerate(segments) if segment.name == label]
            if len(indices) == 1:
                return f"(segment {indices[0]})"
            elif indices:
                return "(join {})".format(" ".join(f"(segment {i})" for i in indices))
        return f'"{label}"'


class apical_dendrites(base_morphology.apical_dendrites):

    def get_region(self, morphology=None):
        return f"(tag {SectionType.apical_dendrite.value})"


class basal_dendrites(base_morphology.basal_dendrites):

    def get_region(self, morphology=None):
        return f"(tag {SectionType.basal_dendrite.value})"


class dendrites(base_morphology.dendrites):

    def get_region(self, morphology=None):
        return f"(join (tag {SectionType.apical_dendrite.value}) (tag {SectionType.basal_dendrite.value}))"


class axon(base_morphology.axon):

    def get_region(self, morphology=None):
        return f"(tag {SectionType.axon.value})"


class soma(base_morphology.soma):

    def get_region(self, morphology=None):
        return f"(tag {SectionType.soma.value})"


class random_section(base_morphology.random_section):

    def get_region(self, morphology=None):
        raise NotImplementedError


# --- IonChannelDistributions ---


class HasSelector:

    def get_with_label_selector(self, label):
        return with_label(label)


class uniform(base_morphology.uniform, HasSelector):

    def resolve(self):
        region = self.selector.get_region()
        if isinstance(self.value_provider, (int, float)):
            value = self.value_provider
        else:
            value = self.value_provider.get_value()
        return region, value


class by_distance(base_morphology.by_distance, HasSelector):

    def resolve(self):
        raise NotImplementedError


class by_diameter(base_morphology.by_diameter, HasSelector):

    def resolve(self):
        raise NotImplementedError


# --- LocationGenerators ---


class LabelledLocations(base_morphology.LabelledLocations, HasSelector):

    def generate_locations(self, morphology, label):
        locsets = []
        for location in self.labels:
            if location == "soma":
                # todo: proper location mapping
                # Use the resolved locset expression rather than a label reference
                # ('"root"'): Arbor 0.12's label resolution does not resolve label
                # references in probe locsets against the cell's label_dict.
                locsets.append(('(root)', f"{label}-{location}"))
            elif location == "dendrite":
                locsets.append(('(location 0 0.5)', f"{label}-{location}"))
            elif isinstance(location, str):
                locsets.append((
                    f'(on-components 0.5 (region "{location}"))',
                    f"{label}-{location}"
                ))
        return locsets


class at_distances(base_morphology.at_distances, HasSelector):

    def generate_locations(self, morphology, label):
        # `distal-translate` moves a location that would be beyond the end of a branch
        # to the branch terminal, so we evaluate the locset on the morphology and remove
        # such points: they are the ones also found at a slightly shorter distance.
        # The locations are then given explicitly, which also avoids label references.
        # As with NEURON, a distance on more than one branch gives one label per point.
        from .cells import build_segment_tree

        tree, segment_labels = build_segment_tree(morphology)
        labels = arbor.label_dict({name: expr for name, expr in segment_labels.items() if name})
        cell = arbor.cable_cell(arbor.morphology(tree), arbor.decor(), labels)
        region = _region_expression(self.selector, morphology)

        def points_at(d):
            locset = f"(restrict-to (distal-translate (root) {d!r}) {region})"
            return {(loc.branch, loc.pos) for loc in cell.locations(locset)}

        locations = []
        for d in self.distances:
            points = points_at(d)
            if d > 0:
                points -= points_at(max(d - 1e-3, 0.0))
            # one label per point, as with NEURON, so that each synapse can be a
            # connection target
            location_label = f"{label}-{self.location_label(d)}"
            for i, (branch, pos) in enumerate(sorted(points)):
                locations.append((
                    f"(location {branch} {pos!r})",
                    location_label if len(points) == 1 else f"{location_label}#{i}"
                ))
        return locations


class random_placement(base_morphology.random_placement, HasSelector):

    def generate_locations(self, morphology, label):
        raise NotImplementedError


class centre(base_morphology.centre, HasSelector):

    def generate_locations(self, morphology, label):
        region = _region_expression(self.selector, morphology)
        return [(f'(on-components 0.5 {region})',
                 f"{label}-{self.selector.name}.centre")]
