import math
import unittest


def sphere_volume(n, r):
    """
    Функция для вычисления объема N-мерной сферы.

    n: размерность пространства
    r: радиус сферы
    """
    volume = (math.pi ** (n / 2) / math.gamma(n / 2 + 1)) * r**n
    return volume


class HelloWorldTest(unittest.TestCase):
    def setUp(self):
        self.dims = [0, 1, 2, 3]
        self.resp = [1.0, 2.0, 3.14, 4.19]

    def test_dummy(self):
        for ndim, resp in zip(self.dims, self.resp, strict=False):
            ans = sphere_volume(n=ndim, r=1)
            self.assertAlmostEqual(ans, resp, delta=0.1)


if __name__ == "__main__":
    unittest.main()
