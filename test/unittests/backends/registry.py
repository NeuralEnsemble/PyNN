REG_ATTR = ''

registry = []


def register(exclude=[]):
    def inner_register(scenario):
        setattr(scenario, REG_ATTR, True)
        scenario.exclude = exclude
        return scenario
    return inner_register


def runTest(self):
    pass


def register_class():
    def inner_register(cls):
        cls.registry = []
        if cls not in registry:
            setattr(cls, "runTest", eval("runTest"))
            registry.append(cls)
        for name, func in list(cls.__dict__.items()):
            if hasattr(func, REG_ATTR):
                cls.registry.append(func)
        return cls
    return inner_register
