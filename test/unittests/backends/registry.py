REG_ATTR = ''

registry = []

def register(exclude=[]):
    def inner_register(scenario):
        print "registering %s with exclude=%s" % (scenario, exclude)
        setattr(scenario, REG_ATTR, True)
        scenario.exclude = exclude
        return scenario
    return inner_register

def register_class():
    def inner_register(cls):
        cls.registry = []
        print "registering %s" % (cls)
        if cls not in registry:
            registry.append(cls)
        for name, func in list(cls.__dict__.items()):
            if hasattr(func, REG_ATTR):
                print "name =", name
                cls.registry.append(func)
        return cls
    return inner_register