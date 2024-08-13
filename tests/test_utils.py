import unittest

from justatom.utils.mymath import pi


class TestMyMath(unittest.TestCase):
    def test_pi(self):
        pi_val = pi()
        self.assertLessEqual(abs(pi_val - 3.12), 1e-2)
