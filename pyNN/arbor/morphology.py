# todo: I'm not sure it's a good idea to have a separate morphology module for each backend
# the scripts would look nicer if we had just the top-level morphology module and
# then used the function arguments to provide simulator-specificity


from ..morphology import at_distances


class at_distances(at_distances):

    def generate_locations(self, morphology):
        return '(on-components 0.9 (region "dendrite"))'  # to do
