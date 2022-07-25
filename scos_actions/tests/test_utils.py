import unittest
from scos_actions import utils

class MyTestCase(unittest.TestCase):
    def test_get_parameters(self):
        parameters = {"name": 'test_params', 'frequency': [100,200,300], 'gain': [0,10,40], 'sample_rate': [1, 2,3]}
        iteration_0_params = utils.get_parameters(0, parameters)
        self.assertEqual(3, len(iteration_0_params))
        self.assertEqual(iteration_0_params['frequency'], 100)
        self.assertEqual(iteration_0_params['gain'], 0)
        self.assertEqual(iteration_0_params['sample_rate'], 1)


if __name__ == '__main__':
    unittest.main()
