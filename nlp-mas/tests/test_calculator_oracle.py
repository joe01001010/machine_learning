"""Independent expected-value checks for the experiment's calculator oracle."""
from pathlib import Path
import tempfile
import unittest

from calculator_oracle import reference


class CalculatorOracleTests(unittest.TestCase):
    def expect(self, expression, expected, kind='rational', **options):
        self.assertEqual(reference(expression, **options), {
            'expected_stdout': expected + '\n', 'expected_exit_code': 0, 'kind': kind})

    def test_precedence_and_both_exponent_spellings(self):
        cases = {'3+4*2-1': '10', '(2+3)*4': '20', '2^3^2': '512',
                 '-2^2': '-4', '(-2)**2': '4', '2^-2': '0.25',
                 '10-3-2': '5', '12/3*2': '8'}
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.expect(expression, expected)

    def test_decimal_literals_are_exact_not_binary_floats(self):
        self.expect('0.1+0.2', '0.3')
        self.expect('1.00000000000000001-1', '0.00000000000000001')
        self.expect('1e-20 + 2e-20', '0.00000000000000000003')
        self.expect('.5 + 2.', '2.5')

    def test_rational_decimals_and_one_full_repeat_cycle(self):
        cases = {'1/3': '0.3', '1/6': '0.16', '1/7': '0.142857',
                 '10/3': '3.3', '-1/6': '-0.16', '1/12': '0.083',
                 '22/7': '3.142857', '1/90': '0.01', '2/11': '0.18',
                 '1/8': '0.125', '100/4': '25', '0/7': '0'}
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.expect(expression, expected)

    def test_negative_floor_division_and_modulo(self):
        cases = {'-7//3': '-3', '-7%3': '2', '7//-3': '-3', '7%-3': '-2',
                 '5.5%2': '1.5', '5.5//2': '2'}
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.expect(expression, expected)

    def test_exact_roots_stay_rational(self):
        self.expect('9^0.5', '3')
        self.expect('8**(1/3)', '2')
        self.expect('(1/4)^.5', '0.5')
        self.expect('(2^.5+1)*(2^.5-1)', '1')
        self.expect('(2^.5+1)*(2^.5-1)+1e-30', '1.000000000000000000000000000001')

    def test_irrationals_use_17_significant_digits(self):
        self.expect('2^0.5', '1.414213562373095', 'irrational')
        self.expect('10**.3', '1.9952623149688796', 'irrational')
        self.expect('3**.3', '1.3903891703159093', 'irrational')
        self.expect('-2^0.5', '-1.414213562373095', 'irrational')
        self.expect('2^0.5', '1.4142', 'irrational', significant_digits=5)

    def test_rounding_resolves_tiny_perturbation_near_half_even_boundary(self):
        self.expect('(2.1025+1e-200)^.5', '1.5', 'irrational', significant_digits=2)
        self.expect('(2.1025-1e-200)^.5', '1.4', 'irrational', significant_digits=2)

    def test_invalid_syntax_domain_and_unsupported_code(self):
        for expression in ('', '2+', '1/0', '1//0', '1%0', '0^-1', '(-1)^.5',
                           'True+1', '1j', '[1,2]', '(1,2)', 'abs(-2)',
                           '(1).__class__', 'lambda: 3', '0x10', 'nan', 'inf'):
            with self.subTest(expression=expression):
                self.assertEqual(reference(expression), {
                    'expected_stdout': 'Error: invalid input\n', 'expected_exit_code': 1, 'kind': 'error'})

    def test_malicious_expression_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'should-not-exist.txt'
            expression = f"__import__('pathlib').Path({str(path)!r}).write_text('executed')"
            self.assertEqual(reference(expression)['kind'], 'error')
            self.assertFalse(path.exists())

    def test_limits_raise_instead_of_inventing_invalid_mathematics(self):
        expressions = ['1'*2001, '9'*201, '2**1000000', '2**(1/1000000)',
                       '1e1000000', '+'.join(['1']*100), '1/10007',
                       '(' * 300 + '1' + ')' * 300]
        for expression in expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    reference(expression)
        with self.assertRaises(ValueError):
            reference('2**(2^.5)')

    def test_invalid_precision_configuration(self):
        for digits in (0, 101, True, 1.5):
            with self.subTest(digits=digits):
                with self.assertRaises(ValueError):
                    reference('2^.5', significant_digits=digits)


if __name__ == '__main__':
    unittest.main()
