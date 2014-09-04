
registry = []

def register(exclude=[], include_only=[]):
    def inner_register(scenario):
        #print("registering %s with exclude=%s" % (scenario, exclude))
        if scenario not in registry:
            scenario.exclude = exclude
            scenario.include_only = include_only
            registry.append(scenario)
        return scenario
    return inner_register