from math import prod
import sys

expression = sys.argv[1]
totals = [sum(map(int, group.split('+'))) for group in expression.split('*')]
print(prod(totals))
