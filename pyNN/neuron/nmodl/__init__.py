"""

"""

from neuron import h
from pyNN.models import BaseIonChannelModel
from pyNN.morphology import IonChannelDistribution
from .. import simulator


class NMODLIonChannelModel(BaseIonChannelModel):
    conductance_density_parameter=None

    def __init__(self, **parameters):
        mechanism = getattr(simulator.dummy(0.5), self.name)
        # now get list of range variables (dir(mechanism), ends with _<name>)
        lname = len(self.name)
        self.range_variables = [var[:-lname-1] for var in dir(mechanism) if var.endswith("_" + self.name)]
        # the problem now is that only some of the range variables are parameters
        # is there any way to introspect which these are?
        # maybe check the NEURON GUI - does it limit the recordable vars?
        # otherwise we either require the source, and parse the NMODL
        # or we require the user to tell us
        if self.conductance_density_parameter is None:
            for varname in ("gbar", "gnabar", "gkbar", "gcabar", "g", "gmax"):
                if varname in self.range_variables:
                    self.conductance_density_parameter = varname
                    break
        BaseIonChannelModel.__init__(self, **parameters)


    def translate(self, parameters):       
        return parameters

    def get_schema(self):
        #return {
        #    self.conductance_density_parameter: IonChannelDistribution
        #}
        return {varname: IonChannelDistribution
                for varname in self.range_variables}


def NMODLChannel(name, conductance_density_parameter=None):
    simulator.dummy.insert(name)
    return type(name, 
                (NMODLIonChannelModel,),
                {"name": name,
                 "model": name,
                 "conductance_density_parameter": conductance_density_parameter})
