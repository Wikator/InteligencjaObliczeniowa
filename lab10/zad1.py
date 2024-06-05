import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt
import numpy as np
import math

# a)

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)

optimizer.optimize(fx.sphere, iters=1000)

# b)

x_max = [2, 2]
x_min = [1, 1]
my_bounds = (x_min, x_max)

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=my_bounds)

optimizer.optimize(fx.sphere, iters=1000)

# Wynik jest bardzo podobny

# c)

x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)

# d) e)

def endurance(x, y, z, u, v, w):
    return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w)

def f(x):
    n_particles = x.shape[0]
    j = [endurance(x[i][0], x[i][1], x[i][2], x[i][3], x[i][4], x[i][5]) for i in range(n_particles)]
    return np.array(j)

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)
optimizer.optimize(lambda x: -f(x), iters=1000)

# f)

plot_cost_history(optimizer.cost_history)
plt.show()
