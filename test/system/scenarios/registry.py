from testconfig import config

registry = []


def register(exclude=[]):
    def inner_register(scenario):
        if scenario not in registry and not ('testName' in config and not scenario.__name__ == config['testName']):
            scenario.exclude = exclude
            registry.append(scenario)
        return scenario
    return inner_register
