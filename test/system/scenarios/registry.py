
registry = []

def register(exclude=[]):
    def inner_register(scenario):
        if scenario not in registry:
            scenario.exclude = exclude
            registry.append(scenario)
        return scenario
    return inner_register