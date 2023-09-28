"""

"""

from warnings import warn
import numpy as np
import arbor

from .. import common, errors
from ..standardmodels import StandardCellType
from ..parameters import ParameterSpace, simplify, Sequence
from . import simulator
from .recording import Recorder


class Assembly(common.Assembly):
    _simulator = simulator


class PopulationView(common.PopulationView):
    _assembly_class = Assembly
    _simulator = simulator

    def _get_parameters(self, *names):
        if isinstance(self.celltype, StandardCellType):
            if any(name in self.celltype.computed_parameters() for name in names):
                # need all parameters in order to calculate values
                native_names = self.celltype.get_native_names()
            else:
                native_names = self.celltype.get_native_names(*names)
            native_parameter_space = self._get_native_parameters(*native_names)
            parameter_space = self.celltype.reverse_translate(native_parameter_space)
        else:
            parameter_space = self._get_native_parameters(*native_names)
        return parameter_space

    def _get_native_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            value = self.parent._parameters[name]
            if isinstance(value, np.ndarray):
                value = value[self.mask]
            parameter_dict[name] = simplify(value)
        return ParameterSpace(parameter_dict, shape=(self.size,))  # or local size?

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        for name, value in parameter_space.items():
            try:
                self.parent._parameters[name][self.mask] = value.evaluate(simplify=True)
            except ValueError:
                raise errors.InvalidParameterValueError(
                    f"{name} should not be of type {type(value)}")

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)


class Population(common.Population):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        super().__init__(size, cellclass, cellparams, structure, initial_values, label)
        self._simulator.state.network.add_population(self)

    @property
    def arbor_cell_kind(self):
        return self.celltype.arbor_cell_kind

    def arbor_cell_description(self, gid):
        index = self.id_to_index(gid)
        if self.celltype.arbor_cell_kind == arbor.cell_kind.spike_source:
            cell_descr = self._arbor_cell_description
            if not cell_descr._evaluated:
                cell_descr.evaluate()
            params = list(cell_descr)[index]  # inefficient to do this for every gid, need to fix
            for key, value in params.items():
                if isinstance(value, Sequence):
                    params[key] = value.value
            schedule = self.celltype.arbor_schedule(**params)
            return arbor.spike_source_cell("spike-source", schedule)
        else:
            args = self._arbor_cell_description[index]
            return arbor.cable_cell(args["tree"], args["decor"], args["labels"])

    def _create_cells(self):
        # for now, we create all cells and store them in memory,
        # however it might be better to override __getitem__
        # and create them on the fly

        if isinstance(self.celltype, StandardCellType):
            parameter_space = self.celltype.native_parameters
        else:
            parameter_space = self.celltype.parameter_space

        parameter_space.shape = (self.size,)

        if self.celltype.arbor_cell_kind == arbor.cell_kind.spike_source:
            self._arbor_cell_description = parameter_space
        else:
            self._arbor_cell_description = parameter_space["cell_description"]
            self._arbor_cell_description.base_value.set_shape(parameter_space.shape)

        id_range = np.arange(simulator.state.id_counter,
                             simulator.state.id_counter + self.size)
        self.all_cells = np.array(
            [simulator.Cell(id) for id in id_range],
            dtype=simulator.Cell
        )

        # for i, cell in enumerate(self.all_cells):
        #     #for key, value in parameter_space.items():
        #     #    setattr(cell, key, value)
        #     cell.morph = parameter_space["tree"]
        #     cell.decor = parameter_space["decor"](i)
        #     cell.labels = parameter_space["labels"]
        #     cell.parent = self
        #     cell.decor.place('"root"', arbor.threshold_detector(-10), f"detector-{cell.gid}")

        self._parameters = parameter_space  # used for querying parameters before/after running simulation

        simulator.state.id_counter += self.size
        self._mask_local = np.ones_like(id_range, dtype=bool)

    def _set_initial_value_array(self, variable, initial_values):
        if variable != "v":
            warn("todo: handle initial values for ion channel states")
            # may have to handle this at the same time as setting parameters
            # it is not clear to me if Arbor supports updating decors
            # after their creation, other than by set_property
            # maybe keep a reference to the return values of arbor.paint?
            return
        self._arbor_cell_description.base_value.set_initial_values(variable, initial_values)

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _get_parameters(self, *names):
        if isinstance(self.celltype, StandardCellType):
            if any(name in self.celltype.computed_parameters() for name in names):
                # need all parameters in order to calculate values
                native_names = self.celltype.get_native_names()
            else:
                native_names = self.celltype.get_native_names(*names)
            native_parameter_space = self._get_native_parameters(*native_names)
            parameter_space = self.celltype.reverse_translate(native_parameter_space)
        else:
            parameter_space = self._get_native_parameters(*native_names)
        return parameter_space

    def _get_native_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            parameter_dict[name] = simplify(self._parameters[name])
        return ParameterSpace(parameter_dict, shape=(self.local_size,))

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        parameter_space.evaluate(simplify=False, mask=self._mask_local)
        for name, value in parameter_space.items():
            self._parameters[name] = value
