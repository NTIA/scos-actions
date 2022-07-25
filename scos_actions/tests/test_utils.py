import unittest
from scos_actions import utils

class MyTestCase(unittest.TestCase):
    def test_get_parameters(self):
        parameters = {"name": 'test_params', 'frequency': [100,200,300], 'gain': [0,10,40], 'sample_rate': [1, 2,3]}
        iteration_params = utils.get_iterable_parameters(parameters)
        self.assertEqual(3, len(iteration_params))
        self.assertEqual(iteration_params[0]['frequency'], 100)
        self.assertEqual(iteration_params[0]['gain'], 0)
        self.assertEqual(iteration_params[0]['sample_rate'], 1)


if __name__ == '__main__':
    unittest.main()
