#!/usr/bin/env python
"""
Script to run doctests.
"""

import doctest
import sys
from optparse import OptionParser

optionflags = doctest.IGNORE_EXCEPTION_DETAIL+doctest.NORMALIZE_WHITESPACE

class MyOutputChecker(doctest.OutputChecker):
    """
    Modification of doctest.OutputChecker to work better with the PyNN
    users' manual:
      * Often, we don't want to have the output that is printed
    by Python in the manual, as it just takes up space without adding any
    useful information.
    """
    
    def __init__(self,strict):
        self.strict = strict
    
    def check_output(self, want, got, optionflags):
        if self.strict:
            return doctest.OutputChecker.check_output(self, want, got, optionflags)
        else:
            if want == '':
                return True
            else:
                try:
                    int(want) and int(got)
                    return True
                except ValueError:
                    return doctest.OutputChecker.check_output(self, want, got, optionflags)

def mytestfile(filename, globs, optionflags, strict=False):
    parser = doctest.DocTestParser()
    if globs is None:
        globs = {}
    else:
        globs = globs.copy()
    name = os.path.basename(filename)
    
    runner = doctest.DocTestRunner(checker=MyOutputChecker(strict=strict), optionflags=optionflags)
    # Read the file, convert it to a test, and run it.
    s = open(filename).read()
    test = parser.get_doctest(s, globs, name, filename, 0)
    runner.run(test)
    runner.summarize()
    return runner.failures, runner.tries

# ==============================================================================
if __name__ == "__main__":
    
    # Process command line
    parser = OptionParser(usage="usage: %prog [options] FILE")
    parser.add_option("-s", "--simulator", dest="simulator",
                      type="choice", choices=('nest1','nest2', 'neuron','oldneuron','pcsim'),
                      help="run doctests with SIMULATOR", metavar="SIMULATOR",
                      default='nest2')
    parser.add_option("--strict", action="store_true", dest="strict", default=False,
                  help="Use the original doctest output checker, not the more lax local version.")

    if 'nrniv' in sys.argv[0]:
        (options, args) = parser.parse_args(sys.argv[5:])
    else:
        (options, args) = parser.parse_args()
    if len(args) == 1:
        docfile = args[0]
    else:
        parser.print_help()
        sys.exit(1)
    
    # Run test
    exec("from pyNN.%s import *" % options.simulator)
    setup(max_delay=1.0,debug=True)
    mytestfile(docfile, globs=globals(), optionflags=optionflags, strict=options.strict)

    sys.exit(0)
