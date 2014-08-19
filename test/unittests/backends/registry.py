REG_ATTR = ''

registry = []

def register(include_only='', exclude=[]):
    def inner_register(scenario):
        setattr(scenario, REG_ATTR, True)
        scenario.exclude = exclude
        scenario.include_only = include_only
        if include_only:
            scenario.exclude = []
        else:
            scenario.exclude = exclude
        print("registering %s with include_only =%s, exclude=%s" % (scenario, scenario.include_only, scenario.exclude))
        return scenario
    return inner_register

def runTest(self):
    pass


def register_class():
    def inner_register(cls):
        cls.registry = []
        print("registering %s" % cls)
        if cls not in registry:
            setattr(cls, "runTest", eval("runTest"))
            registry.append(cls)
        for name, func in list(cls.__dict__.items()):
            if hasattr(func, REG_ATTR):
                #print("name =", name
                cls.registry.append(func)
        return cls
    return inner_register