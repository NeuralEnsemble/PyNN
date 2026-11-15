"""



"""

from collections import defaultdict

import numpy as np
from neuron import h

from .. import morphology as base_morphology


def _root_location(cell):
    """Return the root of the morphology, from which path distances are measured, as a NEURON
    segment: the proximal end of the root section, unless the cell specifies otherwise."""
    if getattr(cell, "root_location", None) is not None:
        section, x = cell.root_location
        return section(x)
    for section in cell.sections.values():
        if section.parentseg() is None:
            return section(0)
    raise ValueError("Cell has no root section")


# --- MorphologyFilters ---


class with_label(base_morphology.with_label):

    def __call__(self, morphology, **kwargs):
        section_index = np.array([], dtype=int)
        labels = list(self.labels)
        for label in labels:
            if label in morphology.section_groups:
                # ids.extend([id(seg) for seg in morphology.section_groups[label]])
                section_index = np.hstack((section_index, morphology.section_groups[label]))
                labels.remove(label)
        if labels:
            matched = set()
            for i, segment in enumerate(morphology.segments):
                if segment.name in labels:
                    section_index = np.hstack((section_index, np.array([i])))
                    matched.add(segment.name)
            labels = [label for label in labels if label not in matched]
        if labels:
            raise ValueError("No sections or groups match label '{}'".format("', '".join(labels)))
        return section_index


class dendrites(base_morphology.dendrites):

    def __call__(self, morphology, filter_by_section=False):
        """Return an index (integer NumPy array) that can be used
        to retrieve the sections corresponding to the filter. """
        from morphio import SectionType

        section_index = np.array([], dtype=int)
        for label in (SectionType.apical_dendrite, SectionType.basal_dendrite):
            if label in morphology.section_groups:
                section_index = np.hstack((section_index, morphology.section_groups[label]))
        if filter_by_section:
            section_index = np.intersect1d(section_index,
                                           np.fromiter(filter_by_section, dtype=int))
        if section_index.size < 1:
            raise Exception("No neurites labelled as dendrites")
        return section_index


class apical_dendrites(base_morphology.apical_dendrites):

    def __call__(self, morphology, filter_by_section=False):
        from morphio import SectionType

        if SectionType.apical_dendrite in morphology.section_groups:
            section_index = morphology.section_groups[SectionType.apical_dendrite]
            if filter_by_section:
                section_index = np.intersect1d(section_index,
                                               np.fromiter(filter_by_section, dtype=int))
            return section_index
        else:
            raise Exception("No neurites labelled as apical dendrite")


class basal_dendrites(base_morphology.basal_dendrites):

    def __call__(self, morphology, filter_by_section=False):
        from morphio import SectionType

        if SectionType.basal_dendrite in morphology.section_groups:
            section_index = morphology.section_groups[SectionType.basal_dendrite]
            if filter_by_section:
                section_index = np.intersect1d(section_index,
                                               np.fromiter(filter_by_section, dtype=int))
            return section_index
        else:
            raise Exception("No neurites labelled as basal dendrite")


class axon(base_morphology.axon):

    def __call__(self, morphology, filter_by_section=False):
        from morphio import SectionType

        if SectionType.axon in morphology.section_groups:
            section_index = morphology.section_groups[SectionType.axon]
            if filter_by_section:
                section_index = np.intersect1d(section_index,
                                               np.fromiter(filter_by_section, dtype=int))
            return section_index
        else:
            raise Exception("No neurites labelled as axon")


class soma(base_morphology.soma):

    def __call__(self, morphology, filter_by_section=False):
        from morphio import SectionType

        if SectionType.soma in morphology.section_groups:
            section_index = morphology.section_groups[SectionType.soma]
            if filter_by_section:
                section_index = np.intersect1d(section_index,
                                               np.fromiter(filter_by_section, dtype=int))
            return section_index
        else:
            raise Exception("No neurites labelled as soma")


class random_section(base_morphology.random_section):

    def __call__(self, morphology, **kwargs):
        section_index = self.f(morphology, **kwargs)
        if len(section_index) < 1:
            raise Exception("List of sections is empty.")
        return [np.random.choice(section_index)]


sample = random_section  # alias


# --- IonChannelDistributions ---


class HasSelector:

    def get_with_label_selector(self, label):
        return with_label(label)


class uniform(base_morphology.uniform, HasSelector):

    def value_in(self, morphology, index):
        if hasattr(self.selector, "labels") and self.selector.labels == ('all',):
            return self.value_provider
        elif hasattr(self.selector, "labels") and self.selector.labels == ('soma',):
            if index == morphology.soma_index:
                return self.value_provider
            else:
                return self.absence
        elif isinstance(self.selector, base_morphology.MorphologyFilter):
            selected_indices = self.selector(morphology)
            if index in selected_indices:
                return self.value_provider
            else:
                return self.absence
        else:
            raise NotImplementedError("selector '{}' not yet supported".format(self.selector))


class by_distance(base_morphology.by_distance, HasSelector):

    def value_in(self, morphology, index):
        distance_function = self.value_provider
        if isinstance(self.selector, base_morphology.MorphologyFilter):
            selected_indices = self.selector(morphology)  # need to cache this, or allow index to be an array
            if index in selected_indices:
                distance = morphology.get_distance(index)
                return distance_function(distance)
            else:
                return self.absence
        else:
            raise NotImplementedError("selector '{}' not yet supported".format(self.selector))


class by_diameter(base_morphology.by_diameter, HasSelector):
    """Distribution as a function of neurite diameter."""

    def value_in(self, morphology, index):
        diameter_function = self.value_provider
        if isinstance(self.selector, base_morphology.MorphologyFilter):
            selected_indices = self.selector(morphology)  # need to cache this, or allow index to be an array
            if index in selected_indices:
                diameter = morphology.get_diameter(index)
                return diameter_function(diameter)
            else:
                return self.absence
        else:
            raise NotImplementedError("selector '{}' not yet supported".format(self.selector))


# --- LocationGenerators ---


class LabelledLocations(base_morphology.LabelledLocations, HasSelector):

    def generate_locations(self, morphology, label_prefix, cell):
        locations = []
        for label in self.labels:
            if label in cell.section_labels:
                section_index = cell.section_labels[label]
                assert len(section_index) == 1, "todo"
                section_id = list(section_index)[0]
                section = cell.sections[section_id]
                location_label = f"{label_prefix}{label}"
                cell.locations[location_label] = Location(section, section_id, 0.5, label=location_label)
                locations.append(location_label)
            else:
                raise ValueError("Cell has no location labelled '{}'".format(label))
        return locations


class at_distances(base_morphology.at_distances, HasSelector):

    def generate_locations(self, morphology, label_prefix, cell):
        tolerance = 1e-4  # µm; NEURON stores 3D points in single precision
        root = _root_location(cell)
        selected = set(self.selector(morphology))
        points = defaultdict(list)
        for section_id in sorted(selected):
            section = cell.sections[section_id]
            parent = section.parentseg()
            parent_selected = parent is not None and any(
                cell.sections[sid] == parent.sec for sid in selected)
            # child sections are connected by their proximal end (x=0)
            start = h.distance(root, section(0))
            for d in self.distances:
                offset = d - start
                if abs(offset) <= tolerance:
                    # a point at the start of the section belongs to the parent section,
                    # unless that is outside the region
                    if not parent_selected:
                        points[d].append((section, section_id, 0.0))
                elif 0 < offset <= section.L + tolerance:
                    points[d].append((section, section_id, min(offset / section.L, 1.0)))
        locations = []
        for d in self.distances:
            location_label = self.location_label(d)
            if label_prefix:
                location_label = f"{label_prefix}-{location_label}"
            for i, (section, section_id, position) in enumerate(points[d]):
                label = location_label if len(points[d]) == 1 else f"{location_label}#{i}"
                cell.locations[label] = Location(section, section_id, position, label=label)
                locations.append(label)
        return locations


class random_placement(base_morphology.random_placement, HasSelector):

    def generate_locations(self, morphology, label_prefix, cell):
        locations = []
        for index, section_id in enumerate(cell.sections):
            density = self.density_function.value_in(morphology, index)
            section = cell.sections[section_id]
            n_synapses = density * section.L
            if n_synapses > 0:
                n_synapses, remainder = divmod(n_synapses, 1)
                rnd = np.random  # todo: use the RNG from the parent Population
                if rnd.uniform() < remainder:
                    n_synapses += 1
            for i in range(int(n_synapses)):
                location_label = f"random-{index}"
                if label_prefix:
                    location_label = f"{label_prefix}-{location_label}"
                cell.locations[location_label] = Location(section, section_id, 0.5, label=location_label)
                # todo: also randomize the position parameter?
                locations.append(location_label)
        return locations


class centre(base_morphology.centre, HasSelector):

    def generate_locations(self, morphology, label_prefix, cell):
        section_index = self.selector(morphology)
        section_id = section_index[len(section_index)//2]
        section = cell.sections[section_id]
        location_label = f"{self.selector.name}.centre"
        if label_prefix:
            location_label = f"{label_prefix}-{location_label}"
        cell.locations[location_label] = Location(section, section_id, 0.5, label=location_label)
        # todo: also randomize the position parameter?
        return [location_label]


center = centre  # for trans-Atlantic compatibility


# --- Location ---

class Location:

    def __init__(self, section, section_id, position, label=""):
        self.section = section
        self.section_id = section_id
        self.position = position
        self.label = label

    def get_section_and_position(self):
        return (self.section, self.section_id, self.position)
