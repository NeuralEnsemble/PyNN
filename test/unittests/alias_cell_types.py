import inspect
from pyNN.standardmodels import StandardCellType

def alias_cell_types(module_name, **cells):
    """
    Replaces sim.CellType by CellType2.
    
    Typically CellType and CellType2 are identical
    
    Example:
    cells = {'IF_cond_exp':sim.IF_cond_exp, 'Toto': sim.IF_cond_exp}
    alias_cell_types(sys.modules[__name__], cells)
    In such a case, in the current module, IF_cond_exp and Toto are aliases of the sim.IF_cond_exp class
    """
    for cell_name in cells:
        setattr(module_name, cell_name, cells[cell_name])
        
def take_all_cell_classes(sim):
    """
    From the simulator sim, it returns a dictionary with all the defined cell ty
    """
    cells = {}
    for name, obj in inspect.getmembers(sim):
        if inspect.isclass(obj):
            if issubclass(obj, StandardCellType):
                cells[name] = obj
    return cells
    