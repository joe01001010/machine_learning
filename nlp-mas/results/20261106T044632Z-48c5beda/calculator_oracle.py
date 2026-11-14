"""Independent arithmetic test oracle; never used by generated applications.

Only an arithmetic AST is accepted. Decimal literals remain exact rationals.
The real-number domain uses Python-style floor division/modulo and principal
powers; a non-real result is an invalid input. Resource limits are oracle
limitations, reported as ValueError rather than invented application behavior.
"""
import ast
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import re

import sympy as sp
from sympy.core.evalf import PrecisionExhausted

MAX_INPUT_CHARACTERS = 2000
MAX_AST_NODES = 160
MAX_AST_DEPTH = 32
MAX_LITERAL_DIGITS = 200
MAX_RATIONAL_BITS = 12000
MAX_POWER_COMPONENT = 4096
MAX_SYMBOLIC_OPERATIONS = 32
MAX_OUTPUT_CHARACTERS = 4096
MAX_EVALUATION_DIGITS = 4096
_NUMBER = re.compile(r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
_ERROR = {'expected_stdout': 'Error: invalid input\n', 'expected_exit_code': 1, 'kind': 'error'}


class _InvalidExpression(Exception):
    pass


def _bounded(value):
    if value.has(sp.zoo, sp.oo, -sp.oo, sp.nan) or value.is_real is False:
        raise _InvalidExpression()
    if value.is_Rational:
        if max(abs(int(value.p)).bit_length(), int(value.q).bit_length()) > MAX_RATIONAL_BITS:
            raise ValueError('Arithmetic result exceeds the oracle integer-size limit')
    elif sp.count_ops(value) > MAX_SYMBOLIC_OPERATIONS:
        raise ValueError('Expression exceeds the oracle symbolic-operation limit')
    return value


def _evaluate(node, source):
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise _InvalidExpression()
        literal = ast.get_source_segment(source, node)
        if literal is None or not _NUMBER.fullmatch(literal):
            raise _InvalidExpression()
        mantissa = re.split('[eE]', literal)[0]
        if sum(character.isdigit() for character in mantissa) > MAX_LITERAL_DIGITS:
            raise ValueError('Numeric literal exceeds the oracle digit limit')
        if re.search('[eE]', literal):
            exponent = re.split('[eE]', literal)[1]
            if len(exponent.lstrip('+-')) > 4 or abs(int(exponent)) > MAX_POWER_COMPONENT:
                raise ValueError('Scientific exponent exceeds the oracle limit')
        try:
            return _bounded(sp.Rational(literal))
        except (TypeError, ZeroDivisionError):
            raise _InvalidExpression() from None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand, source)
        return value if isinstance(node.op, ast.UAdd) else -value
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, (
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
        raise _InvalidExpression()
    left, right = _evaluate(node.left, source), _evaluate(node.right, source)
    if isinstance(node.op, ast.Add):
        value = left + right
    elif isinstance(node.op, ast.Sub):
        value = left - right
    elif isinstance(node.op, ast.Mult):
        value = left * right
    elif isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
        if right == 0:
            raise _InvalidExpression()
        if isinstance(node.op, ast.Div):
            value = left / right
        else:
            quotient = sp.floor(left / right)
            if quotient.is_Integer is not True:
                raise ValueError('The oracle could not establish an exact floor quotient')
            value = quotient if isinstance(node.op, ast.FloorDiv) else left - right * quotient
    else:
        if right.is_Rational is not True:
            raise ValueError('Non-rational exponents exceed the supported oracle domain')
        if max(abs(int(right.p)), int(right.q)) > MAX_POWER_COMPONENT:
            raise ValueError('Exponent exceeds the oracle power limit')
        if left == 0 and right < 0:
            raise _InvalidExpression()
        if left.is_Rational:
            base_bits = max(abs(int(left.p)).bit_length(), int(left.q).bit_length())
            magnitude = max(1, (abs(int(right.p)) + int(right.q) - 1) // int(right.q))
            if base_bits * magnitude > MAX_RATIONAL_BITS:
                raise ValueError('Power exceeds the oracle integer-size limit')
        value = left ** right
    return _bounded(value)


def _rational_text(value):
    numerator, denominator = abs(int(value.p)), int(value.q)
    integer, remainder = divmod(numerator, denominator)
    prefix = ('-' if value < 0 else '') + str(integer)
    if len(prefix) > MAX_OUTPUT_CHARACTERS:
        raise ValueError('Result exceeds the oracle output limit')
    if not remainder:
        return prefix
    digits, seen = [], set()
    while remainder and remainder not in seen:
        if len(prefix) + len(digits) + 2 > MAX_OUTPUT_CHARACTERS:
            raise ValueError('Decimal expansion exceeds the oracle output limit')
        seen.add(remainder)
        digit, remainder = divmod(remainder * 10, denominator)
        digits.append(str(digit))
    # Stopping on the first repeated remainder includes the nonrepeating prefix
    # and exactly one full repeating cycle, without a recurrence marker.
    return prefix + '.' + ''.join(digits)


def _round_significant(value, significant_digits):
    if not value.is_finite():
        raise ValueError('The oracle could not obtain a finite real result')
    if not value:
        return '0'
    adjusted = value.adjusted()
    if abs(adjusted) > MAX_OUTPUT_CHARACTERS:
        raise ValueError('Result exceeds the oracle output limit')
    with localcontext() as context:
        context.prec = max(significant_digits + 20, abs(adjusted) + significant_digits + 20)
        rounded = value.quantize(Decimal(1).scaleb(adjusted - significant_digits + 1),
                                 rounding=ROUND_HALF_EVEN)
        result = format(rounded, 'f')
    if '.' in result:
        result = result.rstrip('0').rstrip('.')
    if len(result) > MAX_OUTPUT_CHARACTERS:
        raise ValueError('Result exceeds the oracle output limit')
    return '0' if result in ('-0', '') else result


def _irrational_text(value, significant_digits):
    # Account for large rational coefficients/cancellation, rather than merely
    # adding a fixed guard to the requested output precision.
    atom_digits = sum((max(abs(int(atom.p)).bit_length(), int(atom.q).bit_length())
                       * 30103) // 100000 + 1 for atom in value.atoms(sp.Rational))
    precision = significant_digits + 80 + atom_digits
    if precision + 40 > MAX_EVALUATION_DIGITS:
        raise ValueError('Expression exceeds the oracle evaluation-precision limit')
    previous = None
    for extra in (0, 40, 120):
        working = precision + extra
        if working > MAX_EVALUATION_DIGITS:
            break
        try:
            approximate = value.evalf(working, maxn=MAX_EVALUATION_DIGITS, strict=True)
        except (PrecisionExhausted, ValueError, TypeError) as exc:
            raise ValueError('The oracle could not establish adequate numerical precision') from exc
        if approximate.is_real is not True or approximate.is_finite is not True:
            raise _InvalidExpression()
        rounded = _round_significant(Decimal(str(approximate)), significant_digits)
        if previous == rounded:
            return rounded
        previous = rounded
    raise ValueError('The oracle could not establish stable significant-digit rounding')


def _parse(source):
    nesting = 0
    for character in source:
        if character == '(':
            nesting += 1
            if nesting > MAX_AST_DEPTH:
                raise ValueError('Expression exceeds the oracle parenthesis-depth limit')
        elif character == ')':
            nesting -= 1
    try:
        tree = ast.parse(source, mode='eval')
    except RecursionError as exc:
        raise ValueError('Expression exceeds the oracle parser-depth limit') from exc
    except (SyntaxError, ValueError) as exc:
        raise _InvalidExpression() from exc
    pending = [(tree, 0)]
    nodes = 0
    while pending:
        node, depth = pending.pop()
        nodes += 1
        if nodes > MAX_AST_NODES or depth > MAX_AST_DEPTH:
            raise ValueError('Expression exceeds the oracle AST size/depth limit')
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return tree


def reference(expression: str, significant_digits: int = 17) -> dict:
    """Return independent expected CLI output, or raise ValueError for limits.

    Syntax/domain errors produce ``Error: invalid input\n`` and exit code 1.
    Exact rational answers retain terminating decimals or one repeating cycle.
    Irrational answers use the requested significant digits and half-even
    rounding. Source text is never evaluated as Python code.
    """
    if not isinstance(expression, str):
        raise ValueError('Expression must be text')
    if type(significant_digits) is not int or not 1 <= significant_digits <= 100:
        raise ValueError('significant_digits must be an integer from 1 to 100')
    if len(expression) > MAX_INPUT_CHARACTERS:
        raise ValueError('Expression exceeds the oracle input-length limit')
    source = expression.strip().replace('^', '**')
    try:
        value = _evaluate(_parse(source).body, source)
        if not value.is_Rational:
            value = _bounded(sp.simplify(value))
        if value.is_Rational:
            result, kind = _rational_text(value), 'rational'
        else:
            result, kind = _irrational_text(value, significant_digits), 'irrational'
    except (_InvalidExpression, ZeroDivisionError):
        return dict(_ERROR)
    return {'expected_stdout': result + '\n', 'expected_exit_code': 0, 'kind': kind}


def basic_reference(expression: str) -> dict:
    """Compute expected output only for the initial integer-operand calculator scope."""
    if not isinstance(expression, str) or len(expression) > MAX_INPUT_CHARACTERS:
        raise ValueError('Provide one short arithmetic expression')
    source = expression.strip()
    try:
        tree = _parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                if type(node.value) is not int or not re.fullmatch(r'[0-9]+', ast.get_source_segment(source, node) or ''):
                    raise _InvalidExpression()
            elif not isinstance(node, (ast.Expression, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div)):
                raise _InvalidExpression()
        value = _evaluate(tree.body, source)
    except (_InvalidExpression, ZeroDivisionError):
        raise ValueError('Out of scope: use nonnegative integer operands, parentheses and binary +, -, *, / without division by zero') from None

    denominator = int(value.q)
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    if denominator != 1:
        raise ValueError('Out of scope: the result must have a terminating decimal representation')
    return {'expected_stdout': _rational_text(value) + '\n', 'expected_exit_code': 0, 'kind': 'rational'}
