from pyNN import recording
import unittest
import os
import numpy

TESTFILE = "compatible_output_test.tmp"
TESTFILE_IN = "compatible_output_test.tmp1"

def arrays_almost_equal(a, b, threshold):
    return (abs(a-b) <= threshold).all()

class MockPopulation(object):
    
    def __init__(self):
        self.first_id = 42
        self.dim = (2,2)
    
    def __len__(self):
        return 100

class WriteCompatibleOutputTests(unittest.TestCase):
    
    def create_v_data(self):
        tarr = numpy.arange(0,100.0,0.1).reshape(1000,1)
        self.saved_data = numpy.empty((0,3))
        for id in range(42,52):
            varr = numpy.random.rand(1000,1)
            idarr = id*numpy.ones((1000,1))
            tmp = numpy.concatenate((tarr, varr, idarr), axis=1)
            self.saved_data = numpy.concatenate((self.saved_data, tmp))
            
    def create_gsyn_data(self):
        tarr = numpy.arange(0,100.0,0.1).reshape(1000,1)
        self.saved_data = numpy.empty((0,4))
        for id in range(42,52):
            gEarr = numpy.random.rand(1000,1)
            gIarr = numpy.random.rand(1000,1)
            idarr = id*numpy.ones((1000,1))
            tmp = numpy.concatenate((tarr, gEarr, gIarr, idarr), axis=1)
            self.saved_data = numpy.concatenate((self.saved_data, tmp))
            
    def create_spike_data(self):
        tarr = numpy.random.uniform(0, 1000.0, (1000, 1))
        idarr = numpy.random.uniform(42, 52, (1000, 1)).astype(int) 
        self.saved_data = numpy.concatenate((tarr, idarr), axis=1)
    
    def tearDown(self):
        for filename in TESTFILE, TESTFILE_IN:
            if os.path.exists(filename):
                os.remove(filename)
        self.header = []
    
    def _test_write_read(self, data_source, variable, population, input_format, expected_data, tolerance=1e-12):
        
        recording.write_compatible_output1(data_source=data_source,
                                           user_filename=TESTFILE,
                                           variable=variable,
                                           input_format=input_format,
                                           population=population,
                                           dt=0.1)
        f = open(TESTFILE)
        self.header = [line for line in f if line[0] == "#"]
        f.close()
        loaded_data = numpy.loadtxt("compatible_output_test.tmp")
        if population is not None:
            loaded_data[:,-1] += population.first_id
        self.assertEqual(expected_data.shape, loaded_data.shape)
        if arrays_almost_equal(expected_data, loaded_data, tolerance):
            assert True
        else:
            print "First row: %s <--> %s" % (expected_data[0,:], loaded_data[0,:])
            print "Last row: %s <--> %s" % (expected_data[-1,:], loaded_data[-1,:])
            diff = abs(expected_data - loaded_data)
            print "Max difference: %g" % diff.max()
            assert False
            
    def test_v_from_array_noPop(self):
        """
        Output format should be v id
        """
        self.create_v_data()
        self._test_write_read(self.saved_data, 'v', None, 't v id', self.saved_data[:, 1:]) # cols 1 and 2
        self.assertEqual(self.header, ['# dt = 0.1\n', '# n = 10000\n'])
        
    def test_v_from_array_withPop(self):
        self.create_v_data()
        self._test_write_read(self.saved_data, 'v', MockPopulation(), 't v id', self.saved_data[:, 1:])
        self.assertEqual(self.header, ['# first_id = 0\n', '# n = 10000\n', '# dt = 0.1\n', '# dimensions = [2, 2]\n', '# last_id = 99\n'])
            
    def test_spikes_from_array_withPop(self):
        self.create_spike_data()
        self._test_write_read(self.saved_data, 'spikes', MockPopulation(), 't id', self.saved_data, tolerance=1e-9) # both cols
        self.assertEqual(self.header, ['# first_id = 0\n', '# n = 1000\n', '# dt = 0.1\n', '# dimensions = [2, 2]\n', '# last_id = 99\n'])
        
    def test_gsyn_from_array_withPop(self):
        self.create_gsyn_data()
        self._test_write_read(self.saved_data, 'conductance', MockPopulation(), 't ge gi id', self.saved_data[:, 1:])
        self.assertEqual(self.header, ['# first_id = 0\n', '# n = 10000\n', '# dt = 0.1\n', '# dimensions = [2, 2]\n', '# last_id = 99\n'])
        
    def test_v_from_file(self):
        self.create_v_data()
        numpy.savetxt(TESTFILE_IN, self.saved_data[:, 1:])
        self._test_write_read(TESTFILE_IN, 'v', MockPopulation(), 't v id', self.saved_data[:, 1:]) # cols 1 and 2
        self.assertEqual(self.header, ['# first_id = 0\n', '# n = 10000\n', '# dt = 0.1\n', '# dimensions = [2, 2]\n', '# last_id = 99\n'])
        
    def test_invalid_variable_name(self):
        self.create_spike_data()
        self.assertRaises(Exception,
                          recording.write_compatible_output1,
                          data_source=self.saved_data,
                          user_filename=TESTFILE,
                          variable="foo",
                          input_format="t v id",
                          population=None,
                          dt=0.1)
        
    def test_nonexistent_file(self):
        assert not os.path.exists("this_file_does_not_exist")
        recording.write_compatible_output1(
                          data_source="this_file_does_not_exist",
                          user_filename=TESTFILE,
                          variable="foo",
                          input_format="t v id",
                          population=None,
                          dt=0.1)
        assert not os.path.exists(TESTFILE)
        
# ==============================================================================
if __name__ == "__main__":
    unittest.main()