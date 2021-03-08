#################################################################################
# This script generates predictions of the exact analytical solution for the 
# Beltrami flow. 
# The library DeepXDE is used for all implementation below.
#################################################################################

import numpy as np
import matplotlib.pyplot as plt 

values = [0.75, 1]
a = 1

x, y, z, t = np.meshgrid(
    np.linspace(0, 1, 10),
    np.linspace(0, 1, 10),
    np.linspace(0, 1, 10),
    np.linspace(0, 1, 10),
)

v = -a*(np.exp(a*y)*np.sin(a*z + d*x) + np.exp(a*x)*np.cos(a*y + d*z))*np.exp(-d**2*t)
w = -a*(np.exp(a*x)*np.sin(a*x + d*y) + np.exp(a*y)*np.cos(a*z + d*x))*np.exp(-d**2*t)
p = -1/2*a**2*(np.exp(2*a*x) + np.exp(2*a*y) + np.exp(2*a*z) + 2*np.exp(a*x + d*y)*np.cos(a*z + d*x)*np.exp(a*(y + z)) + 2*np.exp(a*y + d*z)*np.cos(a*x + d*y)*np.exp(a*(z + x)) + 2*np.exp(a*z + d*x)*np.cos(a*y + d*z)*np.exp(a*(x + y)))*np.exp(-2*d**2*t)


t_index = 0
z_index = 0

ax = plt.axes(projection="3d")
ax.plot_surface(x[:, :, z_index, t_index], y[:, :, z_index, t_index], u[:, :, z_index, t_index])
plt.contour(x[:, :, z_index, t_index], y[:, :, z_index, t_index], p[:, :, z_index, t_index], cmap = 'hsv')
plt.show()