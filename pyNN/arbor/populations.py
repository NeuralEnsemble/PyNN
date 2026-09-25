"""

"""

from warnings import warn
import numpy as np
import arbor
from arbor import units as U

from .. import common, errors
from ..standardmodels import StandardCellType
from ..parameters import ParameterSpace, simplify, Sequence
from . import simulator
from . import _compat
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
            schedule_units = getattr(self.celltype, "arbor_schedule_units", {})
            schedule_params = {}
            for key, value in params.items():
                unit = schedule_units.get(key)
                if unit is None or value is None:
                    schedule_params[key] = value
                elif isinstance(value, (list, tuple, np.ndarray)):
                    schedule_params[key] = [float(v) * unit for v in value]
                else:
                    schedule_params[key] = value * unit
            schedule = self.celltype.arbor_schedule(**schedule_params)
            return arbor.spike_source_cell("spike-source", schedule)
        elif self.celltype.arbor_cell_kind == arbor.cell_kind.lif:
            cell_descr = self._arbor_cell_description
            if not cell_descr._evaluated:
                cell_descr.evaluate()
            params = list(cell_descr)[index]
            if params.get("i_offset", 0.0) != 0.0:
                raise NotImplementedError(
                    "Arbor's native lif_cell (IF_curr_delta) cannot inject a "
                    "constant current, so i_offset must be 0. Use the cable-cell "
                    "IF models for current injection.")
            # The source label ("detector") matches what projections.py uses for
            # injectable presynaptic cells; "syn" is the single delta-synapse target.
            cell = arbor.lif_cell("detector", "syn")
            for name, unit in self.celltype.lif_param_units.items():
                setattr(cell, name, float(params[name]) * unit)
            initial_v = getattr(self, "_lif_initial_v", None)
            if initial_v is not None:
                cell.V_m = float(initial_v._partially_evaluate(index, simplify=True)) * U.mV
            return cell
        else:
            args = self._arbor_cell_description[index]
            return _compat.make_cable_cell(
                args["tree"], args["decor"], args["labels"], args["discretization"])

    def _create_cells(self):
        # for now, we create all cells and store them in memory,
        # however it might be better to override __getitem__
        # and create them on the fly

        if isinstance(self.celltype, StandardCellType):
            parameter_space = self.celltype.native_parameters
        else:
            parameter_space = self.celltype.parameter_space

        parameter_space.shape = (self.size,)

        if self.celltype.arbor_cell_kind in (arbor.cell_kind.spike_source,
                                             arbor.cell_kind.lif):
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
        for obj in self.all_cells:
            obj.parent = self

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
        if self.celltype.arbor_cell_kind == arbor.cell_kind.lif:
            # Native lif cells have no decor; the initial v is applied as V_m when
            # the cell is built in arbor_cell_description.
            if variable == "v":
                self._lif_initial_v = initial_values
            return
        if variable != "v":
            # Receptor/synapse state variables (e.g. "excitatory.gsyn") and ion
            # channel states are initialised to zero by the mechanism's INITIAL
            # block; only a non-default request needs handling, which is not yet
            # supported.
            if "." not in variable:
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
