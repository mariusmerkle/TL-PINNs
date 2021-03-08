#################################################################################
# This script generates predictions of the exact analytical solution for the 
# Kovasznay flow. 
# The library DeepXDE is used for all implementation below.
#################################################################################

import numpy as np
import matplotlib.pyplot as plt 

x, y = np.meshgrid(
    np.linspace(-0.5, 2.0, 100),
    np.linspace(-0.5, 1.5, 100)
)

Re = 20
nu = 1/Re
l = 1/(2*nu) - np.sqrt(1/(4*nu**2) + 4*np.pi**2)

u = 1 - np.exp(l*x)*np.cos(2*np.pi*y)
v = l/(2*np.pi)*np.exp(l*x)*np.sin(2*np.pi*y)
p = 1/2*(1 - np.exp(2*l*x))

ax = plt.axes(projection="3d")
ax.plot_surface(x, y, p)
plt.show()