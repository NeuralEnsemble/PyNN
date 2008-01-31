import nrn
import hoc
h  = hoc.HocObject()
dt = h.dt

h('obfunc new_IClamp() { return new IClamp($1) }')
h('obfunc newlist() { return new List() }')
h('obfunc newvec() { return new Vector($1) }')

xopen = h.xopen
quit = h.quit

class HocError(Exception): pass

def hoc_execute(hoc_commands, comment=None):
    assert isinstance(hoc_commands,list)
    if comment:
        logging.debug(comment)
    for cmd in hoc_commands:
        logging.debug(cmd)
        success = hoc.execute(cmd)
        if not success:
            raise HocError('Error produced by hoc command "%s"' % cmd)

def hoc_comment(comment):
    logging.debug(comment)

def psection(section):
    section.push()
    h.psection()
    h.pop_section()

def init():
    h.dt = dt
    h.finitialize()
    
def run(tstop):
    h('tstop = %g' % tstop)
    h('while (t < tstop) { fadvance() }')

class Vector(object):
    n = 0
    
    def __init__(self,arg=10):
        self.name = 'vector%d' % Vector.n
        Vector.n += 1
        h('objref %s' % self.name)
        if isinstance(arg,int):
            h('%s = new Vector(%d)' % (self.name, arg))
            self.hoc_obj = getattr(h, self.name)
        elif isinstance(arg,list):
            h('%s = new Vector(%d)' % (self.name, len(arg)))
            self.hoc_obj = getattr(h, self.name)
            for i,x in enumerate(arg):
                self.x[i] = x
       
    def __getattr__(self,name):
        return getattr(self.hoc_obj, name)
   
    def __len__(self):
        return self.size()
   
    def __str__(self):
        tmp = self.printf()
        return ''
   
    def __repr__(self):
        tmp = self.printf()
        return ''

    # allow array(Vector())
    # Need Vector().toarray for speed though
    def __getitem__(self,i):
        return self.x[i]
   
    def __setitem__(self,i,y):
        self.x[i] = y

    def __getslice__(self,i,j):
        return [self.x[ii] for ii in xrange(i,j)]

    def __setslice__(self,i,j,y):
        assert(len(y)==j-i)

        iter = y.__iter__()

        for ii in xrange(i,j):
            self.x[ii] = iter.next()

    def tolist(self):
        return [self.x[i] for i in range(int(self.size()))]

    def record(self, section, variable, position=0.5):
        #ref = h.ref(variable)
        #self.hoc_obj.record(ref)
        section.push()
        h('%s.record(&%s(%g))' % (self.name, variable, position))
        h.pop_section()                      


class IClamp(object):
      
    def __init__(self, section, position=0.5, delay=0, dur=0, amp=0):
        assert 0 <= position <= 1
        section.push()
        self.__obj = h.new_IClamp(position)
        h.pop_section()
        
    def __getattr__(self, name):
        if name == "delay":
            return self.__obj.__getattribute__('del')
        elif name in ('amp', 'dur'):
            return self.__obj.name
        else:
            return self.__getattribute__(name)
     
    def __setattr__(self, name, value):
        if name == "delay":
            self.__obj.__setattr__('del', value)
        elif name in ('amp', 'dur'):
            self.__obj.__setattr__(name, value)
        else:
            object.__setattr__(self, name, value)

class ParallelContext(object):
    pass

def open(filename, mode='r'):
    """Return an open File object"""
    pass

class File(object):
    """Hoc file object with Python-like syntax added."""
    pass
