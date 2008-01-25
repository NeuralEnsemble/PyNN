# http://oakwinter.com/code/typecheck/
# easy_install typecheck
import typecheck


exa = 1.0e15
peta = 1.0e12
tera = 1.0e9
mega = 1.0e6
kilo = 1000.0
hecto = 100.0
deca = 10.0
deci = 0.1
centi = 0.01
milli = 0.001
micro = 1.0e-6
nano = 1.0e-9
pico = 1.0e-12
femto = 1.0e-15

class Unit(object):
    def __init__(self,relative=None,factor=1.0,bias=0.0,physical_quantity=None,name=None):
        self.name = name
        self.relative=relative
        self.factor = factor
        self.bias = bias
        self.physical_quantity = physical_quantity

    def __mul__(self,factor):
        return Unit(relative=self,factor=factor)

    __rmul__ = __mul__

    def __add__(self,bias):
        return Unit(relative=self,bias=bias)

    __radd__ = __add__


    #def __repr__(self):
    #    return 'Unit(name='+repr(self.name)+',relative='+repr(self.relative)+\
    #           ',factor='+repr(self.factor)+\
    #           ',bias='+repr(self.bias)+',physical_quantity='+\
    #           repr(self.physical_quantity)+')'

    def __str__(self):
        return '<Unit: '+str(self.name)+' >'

    __repr__ = __str__

UnitType = type(Unit)



V = Unit(physical_quantity='electrical potential',name='Volts')
Volts = V
volts = V

mV = milli*V
mV.name = 'milliVolts'
millivolts = mV

s = Unit(physical_quantity='time',name='seconds')
Seconds = s
seconds = s

ms = milli*s
ms.name = 'milliseconds'

F = Unit(physical_quantity='capacitance',name='Farads')



