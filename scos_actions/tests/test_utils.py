import unittest
from scos_actions import utils

class MyTestCase(unittest.TestCase):
    def test_get_parameters(self):
        parameters = {"name": 'test_params', 'frequency': [100,200,300], 'gain': [0,10,40], 'sample_rate': [1, 2,3]}
        iteration_0_params = utils.get_parameters(0)
        self.assertEqual(3, len(iteration_0_params))


if __name__ == '__main__':
    unittest.main()
