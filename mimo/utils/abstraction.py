from operator import add, sub
from functools import reduce

from mimo.utils.data import islist


class Statistics(tuple):

    def __new__(cls, x):
        return tuple.__new__(Statistics, x)

    def __add__(self, y):
        gsum = lambda x, y: reduce(lambda a, b: list(map(add, a, b)) if islist(x, y) else a + b, [x, y])
        return Statistics(tuple(map(gsum, self, y)))

    def __sub__(self, y):
        gsub = lambda x, y: reduce(lambda a, b: list(map(sub, a, b)) if islist(x, y) else a - b, [x, y])
        return Statistics(tuple(map(gsub, self, y)))

    def __mul__(self, a):
        return Statistics(a * e for e in self)

    def __rmul__(self, a):
        return Statistics(a * e for e in self)
