"""Dummy module to allow us to run pydoc, since the actual hoc module is only
available when running nrniv -python. Copy this to hoc.py when you want to run
pydoc, but don't forget to delete hoc.py AND hoc.pyc afterwards."""

def execute(str):
    """Execute a hoc command. Return 1 on success, 0 on failure."""
    pass
    
