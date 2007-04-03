# coding: utf-8
"""Writes documentation for the API in Wiki format."""

import sys
import pyNN.common
import types, string, re, logging

#-- Set up logging -------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y%m%d-%H%M%S',
                    filename='wikidoc.log',
                    filemode='w')
# define a Handler which writes WARNING messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
# set a format which is simpler for console use
formatter = logging.Formatter('%(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

#-- Define global data ---------------------------------------------------------

exclude = ['__module__','__doc__','__builtins__','__file__','__class__',
                '__delattr__', '__dict__', '__getattribute__', '__hash__',
                '__new__','__reduce__','__reduce_ex__','__repr__','__setattr__',
                '__str__','__weakref__',]
#               'time','types','copy',]

leftquote = re.compile(r"'\b")
leftdblquote = re.compile(r'"\b')

classes = {}
functions = []
data = []

default_arg_fmt  = {'wiki'  : '%s<span style="color:grey;">=%s</span>',
                    'latex' : '%s{\\color{grey}=%s}'}
func_sig_fmt     = {'wiki'  : '%s(<span style="font-weight:normal;">%s</span>)',
                    'latex' : '%s(\\mdseries %s)'}
function_fmt     = {'wiki'  : '\n====<span style="color:#0066ff;">%s</span>====\n',
                    'latex' : '\n\\paragraph*{\\color{brightblue}{%s}}\n'}
method_fmt       = {'wiki'  : '\n====<span style="color:#8888ff;">%s</span>====\n',
                    'latex' : '\n\\paragraph*{\\color{brightblue}{%s}}\n'}
staticmethod_fmt = {'wiki'  : '\n====<span style="color:#0066ff;">%s</span> (static)====\n',
                    'latex' : '\n\\paragraph*{\\color{brightblue}{%s} (static)}\n'}
dict_fmt         = {'wiki'  : "\n\n'''%s''' = {\n",
                    'latex' : '\n\\textbf{%s} = $\\lbrace$\n\n'}
dict_fmt_end     = {'wiki'  : '}\n',
                    'latex' : '$\\rbrace$\n'}
data_element_fmt = {'wiki'  : "\n'''%s''' = %s\n",
                    'latex' : "\n\\textbf{%s} = %s\n"}
table_begin      = {'wiki'  : "{|\n",
                    'latex' : "\\begin{tabular}{lll}\n"}
table_end        = {'wiki'  : "|}\n",
                    'latex' : "\\end{tabular}\n"}
table_row_fmt    = {'wiki'  : "| &nbsp;&nbsp;&nbsp; || %s ||: %s\n|-\n",
                    'latex' : '& %s & :%s\\\\\n'}
category_fmt     = {'wiki'  : '\n==%s==\n',
                    'latex' : '\n\\subsection{%s}\n'}
class_fmt        = {'wiki'  : '\n===<span style="color:green">%s</span>===\n',
                    'latex' : '\n\\subsubsection*{%s}\n'}
horiz_line       = {'wiki'  : '\n----\n',
                    'latex' : ''}

#-- Process command line parameters --------------------------------------------
if len(sys.argv) > 1:
    output = sys.argv[1]
else:
    output = 'wiki'
logging.info('Generating API documentation in %s format' % output)
             
#-- Define functions -----------------------------------------------------------

def _(str):
    """Remove extraneous whitespace."""
    lines = str.strip().split('\n')
    lines = [line.strip() for line in lines]
    return '\n'.join(lines)

def funcArgs(func):
    logging.info('Called funcArgs(%s)' % func)
    if hasattr(func,'im_func'):
        func = func.im_func
    code = func.func_code
    fname = code.co_name
    callargs = code.co_argcount
    args = code.co_varnames[:callargs]
    return "%s(%s)" % (fname, string.join(args,', '))

def func_sig(func):
    """Adapted from http://www.lemburg.com/python/hack.py, by  Marc-André Lemburg
       Returns the signature of a Python function/method as string.
       Keyword initializers are also shown using
       repr(). Representations longer than 100 bytes are truncated.
    """
    logging.info('Called func_sig(%s)' % func)
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
                args[i] = default_arg_fmt[output] % (arg,r)
                i = i + 1
        if code.co_flags & 0x0004: # CO_VARARGS
            args.append('*'+code.co_varnames[callargs])
            callargs = callargs + 1
        if code.co_flags & 0x0008: # CO_VARKEYWORDS
            args.append('**'+code.co_varnames[callargs])
            callargs = callargs + 1
        return func_sig_fmt[output] % (fname,string.join(args,', '))
    except AttributeError:
        logging.warning("%s has no attribute 'func_code'" % func)
        return None

#-- Main block -----------------------------------------------------------------

# gather information from the common module
logging.info("Gathering information from the common module.")
for entry in dir(pyNN.common):
    if entry not in exclude:
        instance = eval('pyNN.common.%s' % entry)
        entry_type = type(instance)
        logging.info('  %-30s %s' % (entry,entry_type))
        if entry_type in [types.ClassType, types.TypeType]:
            classes[entry] = { 'methods': [], 'data': [], 'staticmethods': [] }
            for classentry in dir(instance):
                if classentry not in exclude:
                    classentry_type = type(eval('pyNN.common.%s.%s' % (entry,classentry)))
                    logging.info('    %-28s %s' % (classentry,classentry_type))
                    if classentry_type == types.MethodType:
                        classes[entry]['methods'].append(classentry)
                    elif classentry_type == types.FunctionType:
                        classes[entry]['staticmethods'].append(classentry)
                    else:
                        classes[entry]['data'].append(classentry)
                else:
                    logging.info('    %-28s excluded' % classentry)
        elif  entry_type == types.FunctionType:
            functions.append(entry)
        elif entry_type == types.ModuleType:
            pass
        else:
            data.append(entry)
    else:
        logging.info('  %-30s excluded' % entry)

# output starts here
outputStr = ''
if output == 'latex':
    outputStr += '\definecolor{brightblue}{rgb}{0.0,0.38,1.0}\n'
    outputStr += '\definecolor{paleblue}{rgb}{0.5,0.5,1.0}\n'
    outputStr += '\definecolor{grey}{rgb}{0.5,0.5,0.5}\n'

logging.info("==== DATA ====")
outputStr += category_fmt[output] % "Data"
for element in data:
    instance = eval('pyNN.common.%s' % element)
    if type(instance) == types.DictType:
        outputStr += dict_fmt[output] % element
        outputStr += table_begin[output]
        for k,v in instance.items():
            if output == 'latex':
                v = str(v).replace('{',' $\\lbrace$').replace('}',' $\\rbrace$')
            outputStr += table_row_fmt[output] % (k,v)
        outputStr += table_end[output]
        outputStr += dict_fmt_end[output]
    else:
        outputStr +=  data_element_fmt[output] % (element, instance)
    
logging.info("==== FUNCTIONS ====")
outputStr += category_fmt[output] % "Functions"
for funcname in functions:
    funcinst = eval('pyNN.common.%s' % funcname)
    outputStr += function_fmt[output] % func_sig(funcinst)
    if funcinst.__doc__:
        outputStr += _(funcinst.__doc__.strip())
   
logging.info("==== CLASSES ====")
# sort classes by type:
error_classes = {}
celltype_classes = {}
other_classes = {}
for classname in classes.keys():
    if classname.find('Error') > -1:
        error_classes[classname] = classes[classname]
    elif issubclass(eval('pyNN.common.%s' % classname),pyNN.common.StandardCellType):
        celltype_classes[classname] = classes[classname]
    else:
        other_classes[classname] = classes[classname]

logging.info('Sorting classes...')
logging.info('Error classes:    %s' % ', '.join(error_classes.keys()))
logging.info('Celltype classes: %s' % ', '.join(celltype_classes.keys()))
logging.info('Other classes:    %s' % ', '.join(other_classes.keys()))

# Now iterate through the classes
outputStr += category_fmt[output] % "Classes"
for classes in [celltype_classes, other_classes, error_classes]:
    classlist = classes.keys()
    classlist.sort()
    for classname in classlist:
        outputStr += class_fmt[output] % classname
        docstr = eval('pyNN.common.%s.__doc__' % classname)
        if docstr:
            outputStr += _(docstr)
        for methodname in classes[classname]['methods']:
            methodinst = eval('pyNN.common.%s.%s' % (classname,methodname))
            fs = func_sig(methodinst)
            if fs:
                outputStr += method_fmt[output] % fs
                if methodinst.__doc__:
                    outputStr += _(methodinst.__doc__.strip())
        for methodname in classes[classname]['staticmethods']:
            methodinst = eval('pyNN.common.%s.%s' % (classname,methodname))
            fs = func_sig(methodinst)
            if fs:
                outputStr += staticmethod_fmt[output] % fs
                if methodinst.__doc__:
                    outputStr += _(methodinst.__doc__.strip())
        for element in classes[classname]['data']:
            instance = eval('pyNN.common.%s.%s' % (classname,element))
            if type(instance) == types.DictType:
                outputStr += dict_fmt[output] % element
                if len(instance) > 0:
                    outputStr += table_begin[output]
                    for k,v in instance.items():
                        if output == 'latex':
                            v = str(v).replace('{',' $\\lbrace$').replace('}',' $\\rbrace$')
                            outputStr += table_row_fmt[output] % (k,v)
                        elif output == 'wiki':
                            outputStr += table_row_fmt[output] % ('&quot;%s&quot;' % k,v)
                    outputStr += table_end[output]
                outputStr += dict_fmt_end[output]
            else:
                outputStr +=  data_element_fmt[output] % (element, instance)
                
        outputStr += horiz_line[output]
    
if output == 'latex':
    outputStr = outputStr.replace('_','\_')
    outputStr = outputStr.replace('>','$>$')
    outputStr = outputStr.replace('<','$<$')
    outputStr = leftquote.sub('`',outputStr)
    outputStr = leftdblquote.sub('``',outputStr)

print outputStr