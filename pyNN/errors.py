# encoding: utf-8
"""
Defines exceptions for the PyNN API

    InvalidParameterValueError
    NonExistentParameterError
    InvalidDimensionsError
    ConnectionError
    InvalidModelError
    RoundingWarning
    NothingToWriteError
    InvalidWeightError
    NotLocalError
    RecordingError

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

class InvalidParameterValueError(ValueError):
    """Inappropriate parameter value"""
    pass

class NonExistentParameterError(KeyError):
    """
    Model parameter does not exist.
    """

    def __init__(self, parameter_name, model_name, valid_parameter_names=['unknown']):
        Exception.__init__(self)
        self.parameter_name = parameter_name
        self.model_name = model_name
        self.valid_parameter_names = sorted(valid_parameter_names)

    def __str__(self):
        return "%s (valid parameters for %s are: %s)" % (self.parameter_name,
                                                         self.model_name,
                                                         ", ".join(self.valid_parameter_names))

class InvalidDimensionsError(ValueError):
    """Argument has inappropriate shape/dimensions."""
    pass

class ConnectionError(Exception):
    """Attempt to create an invalid connection or access a non-existent connection."""
    pass

class InvalidModelError(Exception):
    """Attempt to use a non-existent model type."""
    pass

class NoModelAvailableError(Exception):
    """The simulator does not implement the requested model."""
    pass

class RoundingWarning(Warning):
    """The argument has been rounded to a lower level of precision by the simulator."""
    pass

class NothingToWriteError(IOError):
    """There is no data available to write."""
    pass

class InvalidWeightError(ValueError):
    """Invalid value for the synaptic weight."""
    pass

class NotLocalError(Exception):
    """Attempt to access a cell or connection that does not exist on this node (but exists elsewhere)."""
    pass

class RecordingError(Exception): # subclass AttributeError?
    """Attempt to record a variable that does not exist for this cell type."""

    def __init__(self, variable, cell_type):
        self.variable = variable
        self.cell_type = cell_type

    def __str__(self):
        msg = "Cannot record %s from cell type %s. Available variables are %s"
        return msg % (self.variable,
                      self.cell_type.__class__.__name__,
                      ",".join(self.cell_type.recordable))
