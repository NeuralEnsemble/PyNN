"""
For every function, class and method found in the common module, this script
checks that the same function, etc, with the same argument names, exists in the
simulator-specific modules.

Needs to be extended to check that arguments have the same default arguments
(see http://www.faqts.com/knowledge_base/view.phtml/aid/5666 for how to obtain
default values of args).

Andrew P. Davison, CNRS, UNIC, May 2006
$Id$
"""

import re, string, types, getopt, sys, shutil, os
shutil.copy('dummy_hoc.py','hoc.py')
import common, neuron, nest, neuron2, pcsim
os.remove('hoc.py'); os.remove('hoc.pyc')


red     = 0010; green  = 0020; yellow = 0030; blue = 0040;
magenta = 0050; cyan   = 0060; bright = 0100
try:
    import ll.ansistyle
    coloured = True
except ImportError:
    coloured = False

# Define some constants
verbose = False
indent = 32
ok = "    ok  "
notfound = "    --  "
inconsistent_args  = "    XX  "
inconsistent_doc   = "   ~ok  "
inconsistency = ""

# Note that we exclude built-ins, modules imported from the standard library,
# and classes defined only in common.
exclude_list = ['__module__','__doc__','__builtins__','__file__','__class__',
                '__delattr__', '__dict__', '__getattribute__', '__hash__',
                '__new__','__reduce__','__reduce_ex__','__repr__','__setattr__',
                '__str__','__weakref__',
                'time','types','copy',
                'InvalidParameterValueError', 'NonExistentParameterError',
                'InvalidDimensionsError', 'ConnectionError',
                'StandardCellType',
                ]

module_list = [neuron, nest, neuron2, pcsim]

if coloured:
    def colour(col,text):
        return str(ll.ansistyle.Text(col,text))
    ok = colour(green,ok)
    inconsistent_args = colour(red,inconsistent_args)
    notfound = colour(yellow+bright,inconsistent_args)
    inconsistent_doc = colour(bright+green,inconsistent_doc)
else:
    def colour(col,text):
        return text

def funcArgs(func):
    if hasattr(func,'im_func'):
        func = func.im_func
    if hasattr(func,'func_code'):
        code = func.func_code
        fname = code.co_name
        callargs = code.co_argcount
        args = code.co_varnames[:callargs]
    else:
        args = []
        fname = func.__name__
    return "%s(%s)" % (fname, string.join(args,','))

def checkDoc(str1,str2):
    """The __doc__ string for the simulator specific classes/functions/methods
    must match that for the common definition at the start. Further information
    can be added at the end."""
    global inconsistency
    if str1 and str2:
        str1 = ' '.join(str1.strip().split()) # ignore differences in white space
        str2 = ' '.join(str2.strip().split())
        nchar1 = len(str1)
        if nchar1 <= len(str2) and str2[0:nchar1] == str1:
            retstr = ok
        else:
            retstr = inconsistent_doc
            inconsistency += "    [" + str1.replace("\n","") + "]\n" + colour(magenta,"    [" + str2.replace("\n","") + "]") + "\n"
    else:
        retstr = inconsistent_doc
        inconsistency += colour(bright+magenta,'    [Missing]') + "\n"
    return retstr

def checkFunction(func):
    """Checks that the functions have the same names, argument names, and
    __doc__ strings."""
    str = ""
    common_args = funcArgs(func)
    common_doc  = func.__doc__
    for module in module_list:
        if dir(module).__contains__(func.func_name):
            modfunc = getattr(module,func.func_name)
            module_args = funcArgs(modfunc)
            if common_args == module_args:
                module_doc = modfunc.__doc__
                str += checkDoc(common_doc,module_doc)
            else:
                str += inconsistent_args
                if verbose: str += common_args + "!=" + module_args
        else:
            str += notfound
    return str

def checkClass(classname):
    """Checks that the classes have the same method names and the same
    __doc__ strings."""
    str = ""
    common_doc  = getattr(common,classname).__doc__
    for module in module_list:            
        if dir(module).__contains__(classname):
            module_doc = getattr(module,classname).__doc__
            str += checkDoc(common_doc,module_doc)
        else:
            str += notfound
    return str

def checkMethod(meth,classname):
    """Checks that the methods have the same names, argument names, and
    __doc__ strings."""
    str = ""
    common_args = funcArgs(meth.im_func)
    common_doc  = meth.im_func.__doc__
    for cls in [getattr(m,classname) for m in module_list]:
        if dir(cls).__contains__(meth.im_func.func_name):
            modulemeth = getattr(cls,meth.im_func.func_name)
            module_args = funcArgs(modulemeth)
            module_doc  = modulemeth.im_func.__doc__
            if common_args == module_args:
                str += checkDoc(common_doc,module_doc)
            else:
                str += inconsistent_args + common_args + "!=" + module_args
        else:
            str += notfound
    return str

def checkStaticMethod(meth,classname):
    """Checks that the methods have the same names, argument names, and
    __doc__ strings."""
    str = ""
    common_args = funcArgs(meth)
    common_doc  = meth.__doc__
    for cls in [getattr(m,classname) for m in module_list]:
        if dir(cls).__contains__(meth.func_name):
            modulemeth = getattr(cls,meth.func_name)
            module_args = funcArgs(modulemeth)
            module_doc = modulemeth.__doc__
            if common_args == module_args:
                str += checkDoc(common_doc,module_doc)
            else:
                str += inconsistent_args + common_args + "!=" + module_args
        else:
            str += notfound
    return str

def checkData(varname):
    """Checks that all modules contain data items with the same name."""
    str = ""
    for module in module_list:
        if dir(module).__contains__(varname):
            str += ok
        else:
            str += notfound
    return str

# Main script

if __name__ == "__main__":
    
    # Parse command line arguments
    verbose = False
    try:
        opts, args = getopt.getopt(sys.argv[1:],"v")
        for opt, arg in opts:
            if opt == "-v":
                verbose = True
    except getopt.GetoptError:
        print "Usage: python testAPI.py [options]\n\nValid options: -v  : verbose output"
        sys.exit(2)

    header = "   ".join(m.__name__.upper() for m in module_list)
    print "\n%s%s" % (" "*(indent+3),header)
    exclude_pattern = re.compile('^' + '$|^'.join(exclude_list) + '$')
    for item in dir(common):
        if not exclude_pattern.match(item):
            fmt = "%s-%ds " % ("%",indent)
            line = ""
            fm = getattr(common,item)
            if type(fm) == types.FunctionType:
                line += colour(yellow,fmt % item) #+ '(function)    '
                line += checkFunction(fm)
            elif type(fm) == types.ClassType or type(fm) == types.TypeType:
                line += colour(cyan,fmt % item) #+ '(class)       '
                line += checkClass(item)
                if line: print line
                if verbose:
                    if inconsistency: print inconsistency.strip("\n")
                    inconsistency = ""
                for subitem in dir(fm):
                    if not exclude_pattern.match(subitem):
                        line = ""
                        fmt = "  %s-%ds " % ("%",(indent-2))
                        fm1 = getattr(fm,subitem)
                        if type(fm1) == types.MethodType:
                            line += colour(yellow,fmt % subitem) #+ '(method)      '
                            line += checkMethod(fm1,item)
                        elif type(fm1) == types.FunctionType:
                            line += colour(yellow+bright,fmt % subitem) #+ '(staticmethod)'
                            line += checkStaticMethod(fm1,item)
                        #else: # class data, should add check
                        if line: print line
                        if verbose:
                            if inconsistency: print inconsistency.strip("\n")
                            inconsistency = ""
            else: # data
                line = colour(bright+red,fmt % item) #+ '(data)        '
                line += checkData(item)
            if line: print line
            if verbose:
                if inconsistency: print inconsistency.strip("\n")
                inconsistency = ""
    print "\n%s%s" % (" "*(indent+3),header)
        
