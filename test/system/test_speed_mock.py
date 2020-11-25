try:
    import unittest2 as unittest
except ImportError:
    import unittest

# import pyNN.nest as sim
import pyNN.mock as sim

from pyNN.utility import Timer


class PopulationTest(unittest.TestCase):
    @staticmethod
    def do_scaling_per_population_test(N):
        timer = Timer()
        timer.start()
        sim.setup()
        timer.mark("setup")
        p = sim.Population(N, sim.IF_curr_exp())
        timer.mark("Population: " + str(N))
        sim.end()
        timer.mark("end")
        elapsed_time = timer.elapsed_time()
        relative_elapsed_time = elapsed_time / N
        print(
            "Creating a {}-sized population took {}s ({}s per neuron)".format(
                N, elapsed_time, relative_elapsed_time
            )
        )

    def test_scaling_per_population(self, sim=sim):
        for powerN in range(13):
            N = 2 ** powerN
            with self.subTest(N=N):
                self.do_scaling_per_population_test(N)

    @staticmethod
    def do_scaling_test(N):
        timer = Timer()
        timer.start()
        sim.setup()
        timer.mark("setup")
        for _ in range(N):
            p = sim.Population(1, sim.IF_curr_exp())
            timer.mark("Population: " + str(N))
        sim.end()
        timer.mark("end")
        elapsed_time = timer.elapsed_time()
        relative_elapsed_time = elapsed_time / N
        print(
            "Creating {} populations took {}s ({}s per population)".format(
                N, elapsed_time, relative_elapsed_time
            )
        )

    def test_scaling(self, sim=sim):
        for powerN in range(13):
            N = 2 ** powerN
            with self.subTest(N=N):
                self.do_scaling_test(N)

    @staticmethod
    def _add_dummy_parameters(celltype, M):
        for i in range(M):
            pname = "tmp{}".format(i)
            celltype.default_parameters[pname] = 0.0
            celltype.units[pname] = "mV"
            celltype.translations[pname] = {
                "translated_name": pname.upper(),
                "forward_transform": pname,
                "reverse_transform": pname.upper(),
            }

    @staticmethod
    def _remove_dummy_parameters(celltype, M):
        for i in range(M):
            pname = "tmp{}".format(i)
            del celltype.default_parameters[pname]
            del celltype.units[pname]
            del celltype.translations[pname]

    @staticmethod
    def do_scaling_cellparams_test(N, M):
        # copy.deepcopy doesn't help here => we add and restore manually
        celltype = sim.IF_cond_exp
        assert len(celltype.get_parameter_names()) < 100
        PopulationTest._add_dummy_parameters(celltype, M)
        sim.setup()
        t0 = Timer()
        t0.start()
        pop = sim.Population(N, celltype())  # this is the culprit
        elapsed_time = t0.elapsed_time()
        relative_elapsed_time = elapsed_time / M
        print(
            "Creating a population with {} parameters took {} ({} per parameter)".format(
                len(celltype.get_parameter_names()), elapsed_time, relative_elapsed_time
            )
        )
        sim.end()
        PopulationTest._remove_dummy_parameters(celltype, M)

    def test_scaling_cellparams(self, sim=sim):
        for powerN in range(16):
            N = 2 ** powerN
            with self.subTest(N=N):
                self.do_scaling_cellparams_test(1, N)

    @staticmethod
    def do_scaling_cellparams_popview_test(N, M):
        # copy.deepcopy doesn't help here => we add and restore manually
        celltype = sim.IF_cond_exp
        assert len(celltype.get_parameter_names()) < 100
        PopulationTest._add_dummy_parameters(celltype, M)
        sim.setup()
        pop = sim.Population(N, celltype())
        pview = sim.PopulationView(pop, [0])
        post_cell = sim.simulator.ID(pview.first_id)
        post_cell.parent = pop  # this is the culprit
        t0 = Timer()
        t0.start()
        post_index = pview.id_to_index(post_cell)
        elapsed_time = t0.elapsed_time()
        relative_elapsed_time = elapsed_time / M
        print(
            "Calling id_to_index on a view into a population with {} parameters took {} ({} per parameter)".format(
                len(celltype.get_parameter_names()), elapsed_time, relative_elapsed_time
            )
        )
        sim.end()
        PopulationTest._remove_dummy_parameters(celltype, M)

    def test_scaling_cellparams_popview(self, sim=sim):
        for powerN in range(8):
            N = 2 ** powerN
            with self.subTest(N=N):
                self.do_scaling_cellparams_popview_test(1, N)


class ProjectionTest(unittest.TestCase):
    @staticmethod
    def do_scaling_per_projection_test(N):
        sim.setup()
        pre = sim.Population(N, sim.IF_cond_exp())
        post = sim.Population(N, sim.IF_cond_exp())

        timer = Timer()
        timer.start()
        proj = sim.Projection(
            pre, post, sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=1.0)
        )
        timer.mark("Projection: " + str(N))
        elapsed_time = timer.elapsed_time()
        relative_elapsed_time = elapsed_time / N

        sim.end()
        print(
            "Creating {}-sized projection took {}s ({}s per synapse)".format(
                N, elapsed_time, relative_elapsed_time
            )
        )

    def test_scaling_per_projection(self, sim=sim):
        for powerN in range(13):
            N = 2 ** powerN
            with self.subTest(N=N):
                self.do_scaling_per_projection_test(N)

    @staticmethod
    def do_scaling_test(N):
        sim.setup()
        pre = sim.Population(N, sim.IF_cond_exp())
        post = sim.Population(N, sim.IF_cond_exp())

        timer = Timer()
        timer.start()
        for i in range(N):

            # FIXME: add check for class-vs-instance in connector, it fails if the connector is a class
            proj = sim.Projection(
                pre[i : i + 1],
                post[i : i + 1],
                sim.OneToOneConnector(),
                synapse_type=sim.StaticSynapse(weight=1.0),
            )
            timer.mark("Projection: " + str(N))
        elapsed_time = timer.elapsed_time()
        relative_elapsed_time = elapsed_time / N

        sim.end()

        print(
            "Creating {} projections took {}s ({}s per projection)".format(
                N, elapsed_time, relative_elapsed_time
            )
        )

    def test_scaling(self, sim=sim):
        for powerN in range(13):
            N = 2 ** powerN
            with self.subTest(N=N):
                self.do_scaling_test(N)


# import profile
if __name__ == "__main__":
    unittest.main()
    # profile.run('print(PopulationTest.do_scaling_cellparams_popview_test(1, 100))')
