# todo: I'm not sure it's a good idea to have a separate morphology module for each backend
# the scripts would look nicer if we had just the top-level morphology module and
# then used the function arguments to provide simulator-specificity

from morphio import SectionType
from .. import morphology as base_morphology



# --- MorphologyFilters ---

class with_label(base_morphology.with_label):

    def get_region(self):
        if len(self.labels) > 1:
            raise NotImplementedError
        assert len(self.labels) != 0
        return f'"{self.labels[0]}"'


class apical_dendrites(base_morphology.apical_dendrites):

    def get_region(self):
        return f"(tag {SectionType.apical_dendrite.value})"


class basal_dendrites(base_morphology.basal_dendrites):

    def get_region(self):
        return f"(tag {SectionType.basal_dendrite.value})"


class dendrites(base_morphology.dendrites):

    def get_region(self):
        return f"(join (tag {SectionType.apical_dendrite.value}) (tag {SectionType.basal_dendrite.value}))"


class axon(base_morphology.axon):

    def get_region(self):
        return f"(tag {SectionType.axon.value})"


class random_section(base_morphology.random_section):

    def get_region(self):
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
                locsets.append(('"root"', f"{label}-{location}"))
            elif location == "dendrite":
                locsets.append(('"mid-dend"', f"{label}-{location}"))
            elif isinstance(location, str):
                locsets.append((
                    f'(on-components 0.5 (region "{location}"))',
                    f"{label}-{location}"
                ))
        return locsets


class at_distances(base_morphology.at_distances, HasSelector):

    def generate_locations(self, morphology, label):

        region = self.selector.get_region()
        locations = [
            (f'(on-components {d} (region {region}))', f"{label}-{region}-{d}")
            for d in self.distances
        ]
        return locations


class random_placement(base_morphology.random_placement, HasSelector):

    def generate_locations(self, morphology, label):
        raise NotImplementedError


class centre(base_morphology.centre, HasSelector):

    def generate_locations(self, morphology, label):
        region = self.selector.get_region()
        return [(f'(on-components 0.5 (region {region}))', f"{label}-{region}-centre")]
