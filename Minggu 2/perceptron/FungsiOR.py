from random import choice
from numpy import array, dot, random
from matplotlib import pyplot as plt

# fungsi matematika


def unit_step(y): return 0 if y < 0 else 1


data_training = [
    (array([0, 0, 1]), 0),
    (array([0, 1, 1]), 1),
    (array([1, 0, 1]), 1),
    (array([1, 1, 1]), 1),
]

w = random.rand(3)
errors = []
teta = 0.2
n = 100

for i in range(n):
    y, expected = choice(data_training)
    result = dot(w, y)
    error = expected - unit_step(result)
    errors.append(error)
    w += teta * error * y

for y, _ in data_training:
    result = dot(y, w)
    print("{}: {} -> {}".format(y[:2], result, unit_step(result)))

plt.ylim([-1, 1])
plt.plot(errors)
plt.show()
