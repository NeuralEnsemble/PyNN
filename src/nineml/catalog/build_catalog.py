"""
For each standard model implemented by the pyNN.nineml backend, export
the NineML description of that model as XML.

"""

from os import path, makedirs
from nineml.abstraction.writers.xml_writer import XMLWriter
import pyNN.nineml
from pyNN.nineml import list_standard_models
from pyNN.nineml.utility import catalog_url


def write_xml(component, directory):
    if not path.exists(directory):
        makedirs(directory)
    XMLWriter.write(component, path.join(directory, "%s.xml" % component.name))


for model_name in list_standard_models():
    model = getattr(pyNN.nineml, model_name)
    cell_component = model.spiking_component_type_to_nineml()    
    write_xml(cell_component, "neurons")
    synapse_component = model.synaptic_receptor_component_type_to_nineml("excitatory")    
    write_xml(synapse_component, "postsynapticresponses")
    

