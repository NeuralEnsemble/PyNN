import units

# one parameter
# unit
# type check
# assign:
# x = Param()
# x = 1,units.mV

# With statements:
# http://www.python.org/dev/peps/pep-0343/

class Param(object):

    def __init__(self,val,unit=None):
        self.val = val
        self.unit=unit
        
    def __ilshift__(self,other):
        self.val = other[0]
        self.unit = other[1]
        return self


# collection of parameters
# handles translation of names, 

class Params(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


ParamsType = type(Params())


# base class for objects with parameters

class ParamObject(object):
    default_params = Params()
    params = Params()

    def __init__(self, params = None, url = None):
        
        # set defaults
        self.params.update(self.default_params)
        if params:
            self.params.update(params)


ParamObjectType = type(ParamObject())
