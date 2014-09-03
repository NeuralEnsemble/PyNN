
registry = []

def register(exclude=[]):
    def inner_register(scenario):
        #print("registering %s with exclude=%s" % (scenario, exclude))
        if scenario not in registry:
            scenario.exclude = exclude
            registry.append(scenario)
        return scenario
    return inner_register