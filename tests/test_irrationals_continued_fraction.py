import numpy as np
from pyoculus.utils import expandcf, fromcf

# Define a list of test cases, each case is a tuple of (NUM, FRAC, CI)
# where NUM is the real number to be expanded into a continued fraction
# where FRAC is a tuple of (numerator, denominator) corresponding to the convergents of NUM with length equal to the length of CI
# and CI is a list of coefficients of the continued fraction expansion
test_cases = [
    (5/7, (5, 7), [0, 1, 2, 2]),
# Content is available under The OEIS End-User License Agreement: http://oeis.org/LICENSE
    # OEIS A010124: Continued fraction for sqrt(19)
    (np.sqrt(19), (1421, 326), [4, 2, 1, 3, 1, 2, 8]),
    # OEIS A001203: Continued fraction for pi
    (np.pi, (833719, 265381), [3, 7, 15, 1, 292, 1, 1, 1, 2]),
    # Golden Mean
    # ((2, 3), [0, 1, 1, 1]),
]

class TestContinuedFractionFunctions:
    def test_expandcf(self):
        for num, frac, ci in test_cases:
            result = expandcf(num, len(ci))
            expected = ci
            np.testing.assert_array_equal(result, expected, f"Failed to correctly expand num={num} into ci={ci}.")

    def test_fromcf(self):
        for num, frac, ci in test_cases:
            result = fromcf(ci)
            expected = frac
            assert result == expected, f"Failed to correctly convert ci={ci} back to fraction frac={frac}."