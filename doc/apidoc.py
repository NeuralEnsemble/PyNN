# coding: utf-8
"""Writes documentation for the API in MediaWiki, Trac or LateX format."""

import sys
import re
from itertools import chain
import types
import pyNN.common
import pyNN.nest
import pyNN.random
import pyNN.utility
import pyNN.recording
from pyNN import __version__
#import pyNN.multisim

simulator = "pyNN.nest"

wiki_format = dict(
    default_arg  = '%s<span style="color:grey;">=%s</span>',
    func_sig     = '%s(<span style="font-weight:normal;">%s</span>)',
    function     = '\n====<span style="color:#0066ff;">%s</span>====\n',
    method       = '\n====<span style="color:#8888ff;">%s</span>====\n',
    staticmethod = '\n====<span style="color:#0066ff;">%s</span> (static)====\n',
    dict         = "\n\n'''%s''' = {\n",
    dict_end     = '}\n',
    data_element = "\n'''%s''' = %s\n",
    table_begin  = "{|\n",
    table_end    = "|}\n",
    table_row    = "| &nbsp;&nbsp;&nbsp; || %s ||: %s\n|-\n",
    title        = '\n=%s=\n',
    section      = '\n==%s==\n',
    class_fmt    = '\n===<span style="color:green">%s</span>===\n',
    horiz_line   = '\n----\n',
)
trac_format = dict(
    default_arg  = '%s=%s',
    func_sig     = '%s(%s)',
    function     = '\n== %s ==\n',
    method       = '\n=== %s ===\n',
    staticmethod = '\n=== %s ===\n',
    dict         = "\n\n'''%s''' = {\n",
    dict_end     = '}\n',
    data_element = "\n'''%s''' = %s\n",
    table_begin  = "\n",
    table_end    = "\n",
    table_row    = "|| '%s':|| %s ||\n",
    title        = '\n= %s =\n----\n',
    section      = '\n----\n= @@%s@@ =\n',
    class_fmt    = '\n== %s ==\n',
    horiz_line   = '\n----\n',
    docstring    = '{{{\n%s\n}}}\n'
)
latex_format = dict(
    default_arg  = '%s{\\color{grey}=%s}',
    func_sig     = '%s(\\mdseries %s)',
    function     = '\n\\paragraph*{\\color{brightblue}{%s}}\n',
    method       = '\n\\paragraph*{\\color{brightblue}{%s}}\n',
    staticmethod = '\n\\paragraph*{\\color{brightblue}{%s} (static)}\n',
    dict         = '\n\\textbf{%s} = $\\lbrace$\n\n',
    dict_end     = '$\\rbrace$\n',
    data_element = "\n\\textbf{%s} = %s\n",
    table_begin  = "\\begin{tabular}{lll}\n",
    table_end    = "\\end{tabular}\n",
    table_row    = '& %s & :%s\\\\\n',
    title        = '\n\\section{%s}\n',
    section      = '\n\\subsection{%s}\n',
    class_fmt    = '\n\\subsubsection*{%s}\n',
    horiz_line   = '',
)
                
exclude = set(['__module__','__doc__','__builtins__','__file__','__class__',
               '__delattr__', '__dict__', '__getattribute__', '__hash__',
               '__new__','__reduce__','__reduce_ex__','__repr__','__setattr__',
               '__str__','__weakref__', '__del__',
              ])
               
leftquote = re.compile(r"'\b")
leftdblquote = re.compile(r'"\b')
camelcase = re.compile(r'(\b([A-Z][a-z]+){2,99})')

def _(str):
    """Remove extraneous whitespace."""
    #lines = str.strip().split('\n')
    #lines = [line.strip() for line in lines]
    lines = [line for line in str.split('\n') if len(line)>0]
    firstline = lines[0]
    pos = 0
    while firstline[pos] == " ":
        pos += 1
    indent = pos
    lines = [line[indent:] for line in lines]
    s = '\n'.join(lines)
    return s.strip()

def funcArgs(func):
    if hasattr(func,'im_func'):
        func = func.im_func
    code = func.func_code
    fname = code.co_name
    callargs = code.co_argcount
    args = code.co_varnames[:callargs]
    return "%s(%s)" % (fname, string.join(args,', '))

def func_sig(func, format):
    """Adapted from http://www.lemburg.com/python/hack.py, by Marc-André Lemburg
       Returns the signature of a Python function/method as string.
       Keyword initializers are also shown using
       repr(). Representations longer than 100 bytes are truncated.
    """
    if hasattr(func,'im_func'): # func is a method
        func = func.im_func
    try:
        code = func.func_code
        fname = code.co_name
        callargs = code.co_argcount
        # XXX Uses hard coded values taken from Include/compile.h
        args = list(code.co_varnames[:callargs])
        if func.func_defaults:
            i = len(args) - len(func.func_defaults)
            for default in func.func_defaults:
                if isinstance(default,float):
                    r = str(default)
                else:
                    try:
                        r = repr(default)
                    except:
                        r = '<repr-error>'
                if len(r) > 100:
                    r = r[:100] + '...'
                arg = args[i]
                if arg[0] == '.':
                    # anonymous arguments
                    arg = '(...)'
                args[i] = format["default_arg"] % (arg,r)
                i = i + 1
        if code.co_flags & 0x0004: # CO_VARARGS
            args.append('*'+code.co_varnames[callargs])
            callargs = callargs + 1
        if code.co_flags & 0x0008: # CO_VARKEYWORDS
            args.append('**'+code.co_varnames[callargs])
            callargs = callargs + 1
        return format["func_sig"] % (fname, ", ".join(args))
    except AttributeError:
        return None

class Document(object):
    
    def __init__(self, title):
        self.title = title
        self.sections = []

    def add_section(self, section):
        self.sections.append(section)

    def render(self, format):
        s = format["title"] % self.title
        s += "\n\n".join([sec.render(format) for sec in self.sections])
        return s

class Section(object):
    
    def __init__(self, title, module):
        self.title = title
        self.module = module
        self.functions = []
        self.classes = []
        
    def add_functions(self, *function_names):
        for name in function_names:
            self.functions.append(Function(name, self.module))
        
    def add_classes(self, *class_names):
        for name in class_names:
            self.classes.append(Class(name, self.module))
            
    def add_abbreviated_classes(self, class_names, staticmethods=[], methods=[], data=[]):
        for name in class_names:
            self.classes.append(AbbreviatedClass(name, self.module, staticmethods, methods, data))
        
    def render(self, format):
        s = format["section"] % self.title
        s += "\n".join([f.render(format) for f in self.functions])
        s += format["horiz_line"]
        s += "\n".join([c.render(format) for c in self.classes])
        return s
        
class Function(object):
    
    def __init__(self, name, module):
        self.name = name
        self.func_obj = eval('%s.%s' % (module, name))
        
    def render(self, format):
        s = format["function"] % func_sig(self.func_obj, format)
        if self.func_obj.__doc__:
            #s += _(self.func_obj.__doc__.strip())
            s += format["docstring"] % _(self.func_obj.__doc__)
        return s
       
class Method(object):
    
    def __init__(self, classname, name, module):
        self.classname = classname
        self.name = name
        self.method_obj = eval('%s.%s.%s' % (module, classname, name))
        
    def render(self, format):
        s = format["method"] % func_sig(self.method_obj, format)
        if self.method_obj.__doc__:
            #s += _(self.method_obj.__doc__.strip())
            s += format["docstring"] % _(self.method_obj.__doc__)
        return s
       
class StaticMethod(object):
    
    def __init__(self, classname, name, module):
        self.classname = classname
        self.name = name
        self.method_obj = eval('%s.%s.%s' % (module, classname, name))
        
    def render(self, format):
        s = format["staticmethod"] % func_sig(self.method_obj, format)
        if self.method_obj.__doc__:
            #s += _(self.method_obj.__doc__.strip())
            s += format["docstring"] % _(self.method_obj.__doc__)
        return s
       
class DictData(object):
    
    def __init__(self, classname, name, module):
        self.classname = classname
        self.name = name
        self.D = eval('%s.%s.%s' % (module, classname, name))
        
    def render(self, format):
        s = format["dict"] % self.name
        s += format["table_begin"]
        for k,v in self.D.items():
            #if output == 'latex':
            #    v = str(v).replace('{',' $\\lbrace$').replace('}',' $\\rbrace$')
            s += format["table_row"] % (k,v)
        s += format["table_end"]
        s += format["dict_end"]
        return s
       
class Class(object):
    
    def __init__(self, name, module):
        self.name = name
        self.class_obj = eval('%s.%s' % (module, name))
        self.methods = []
        self.data = []
        self.staticmethods = []
        for classentry in dir(self.class_obj):
            if classentry not in exclude and (classentry[0] != '_' or classentry[0:2] == '__'): # don't include private methods
                classentry_type = type(eval('%s.%s.%s' % (module,name,classentry)))
                if classentry_type == types.MethodType:
                    self.methods.append(Method(self.name, classentry, module))
                elif classentry_type == types.FunctionType:
                    self.staticmethods.append(StaticMethod(self.name, classentry, module))
                #else:
                #    self.data.append(classentry)
        
    def render(self, format):
        s = format["class_fmt"] % self.name
        if self.class_obj.__doc__:
            s += _(self.class_obj.__doc__)
            #s += format["docstring"] % _(self.class_obj.__doc__)
        for entry in chain(self.staticmethods, self.methods, self.data):
            s += entry.render(format)
        s += format["horiz_line"]
        return s
        
class AbbreviatedClass(Class):
    
    def __init__(self, name, module, staticmethods=[], methods=[], data=[]):
        self.name = name
        self.class_obj = eval('%s.%s' % (module, name))
        self.methods = []
        self.data = []
        self.staticmethods = []
        for classentry in staticmethods:
            self.staticmethods.append(StaticMethod(self.name, classentry))
        for classentry in methods:
            self.methods.append(Method(self.name, classentry, module))
        for classentry in data:
            self.data.append(DictData(self.name, classentry, module))
        
        
def build_document():
    api_doc = Document("PyNN API version %s" % __version__) 
    
    setup = Section("Simulation setup and control", simulator)
    setup.add_functions("setup", "end", "run", "reset", "get_time_step",
                        "get_current_time", "get_min_delay", "get_max_delay",
                        "rank", "num_processes")
    
    lowlevel = Section("Procedural interface for creating, connecting and recording networks",
                       simulator)
    lowlevel.add_functions("create", "connect", "set", "initialize", "record", "record_v",
                           "record_gsyn")
    
    highlevel = Section("Object-oriented interface for creating and recording networks",
                        simulator)
    highlevel.add_classes("Population", "PopulationView", "Assembly")
    space = Section("Classes for defining spatial structure", "pyNN.space")
    space.add_classes("Space")
    space.add_classes("Line", "Grid2D", "Grid3D", "RandomStructure",
                       "Cuboid", "Sphere")
    connect = Section("Object-oriented interface for connecting populations of neurons", simulator)
    connect.add_classes("Projection")
    connect.add_abbreviated_classes(["AllToAllConnector", "OneToOneConnector",
        "FixedProbabilityConnector", "DistanceDependentProbabilityConnector",
        "FixedNumberPreConnector", "FixedNumberPostConnector",
        "FromListConnector", "FromFileConnector", "SmallWorldConnector"],
        #"CSAConnector"],
        methods=["__init__"])
    
    standardcells = Section("Standard neuron models", simulator)
    standardcells.add_abbreviated_classes(["IF_curr_exp", "IF_curr_alpha",
        "IF_cond_exp", "IF_cond_alpha", "EIF_cond_exp_isfa_ista",
        "EIF_cond_alpha_isfa_ista", "IF_facets_hardware1", "HH_cond_exp",
        "SpikeSourcePoisson", "SpikeSourceArray"],
        methods=["__init__"],
        data=["default_parameters"])
        
    
    synapses = Section("Specification of synaptic plasticity", simulator)
    synapses.add_classes("SynapseDynamics", "STDPMechanism")
    synapses.add_abbreviated_classes(["TsodyksMarkramMechanism",
        "AdditiveWeightDependence", "MultiplicativeWeightDependence",
        "AdditivePotentiationMultiplicativeDepression", "GutigWeightDependence",
        "SpikePairRule"], methods=["__init__"])
    
    currentsources = Section("Current injection", simulator)
    currentsources.add_classes("DCSource", "StepCurrentSource", "ACSource", "NoisyCurrentSource")
    
    files = Section("File formats", "pyNN.recording.files")
    files.add_classes("BaseFile")
    files.add_abbreviated_classes(["StandardTextFile", "PickleFile", "NumpyBinaryFile", "HDF5ArrayFile"])
    
    exceptions = Section("Exceptions", "pyNN.errors")
    exceptions.add_abbreviated_classes([
        "InvalidParameterValueError", "NonExistentParameterError",
        "InvalidDimensionsError", "ConnectionError",
        "InvalidModelError", "RoundingWarning", "NothingToWriteError",
        "InvalidWeightError", "NotLocalError", "RecordingError"])
    
    random = Section("The random module", "pyNN.random")
    random.add_classes("NumpyRNG", "GSLRNG", "NativeRNG", "RandomDistribution") 
    
    utility = Section("The utility module", "pyNN.utility")
    utility.add_functions("colour", "notify", "get_script_args", "init_logging")
    utility.add_classes("Timer", "ProgressBar")
    
    #multisim = Section("The multisim module", "pyNN.multisim")
    #multisim.add_classes("MultiSim")
                          
    
    api_doc.add_section(setup)
    api_doc.add_section(highlevel)
    api_doc.add_section(space)
    api_doc.add_section(connect)
    api_doc.add_section(lowlevel)
    api_doc.add_section(standardcells)
    api_doc.add_section(synapses)
    api_doc.add_section(currentsources)
    api_doc.add_section(files)
    api_doc.add_section(exceptions)
    api_doc.add_section(random)
    api_doc.add_section(utility)
    return api_doc

if __name__ == "__main__":
    #-- Process command line parameters --------------------------------------------
    if len(sys.argv) > 1:
        output = sys.argv[1]
    else:
        output = 'trac'
    format = eval("%s_format" % output)
    api_doc = build_document()
    outputStr = api_doc.render(format)
    if output == 'latex':
        outputStr = outputStr.replace('_','\_')
        outputStr = outputStr.replace('>','$>$')
        outputStr = outputStr.replace('<','$<$')
        outputStr = leftquote.sub('`',outputStr)
        outputStr = leftdblquote.sub('``',outputStr)
    if output == 'trac':
        outputStr = outputStr.replace('__','!__')
        outputStr = outputStr.replace('@@','__')
        outputStr = camelcase.sub(r'!\1',outputStr)
    print outputStr