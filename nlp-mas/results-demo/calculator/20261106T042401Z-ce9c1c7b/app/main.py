from math import prod
import sys

expression = sys.argv[1]
totals = [prod(map(int, group.split('*'))) for group in expression.split('+')]
print(sum(totals))
