#################################################################################
# This script generates predictions for all models stored in the database of
# phyiscs-informed neural networks (PINNs) for Burger's equation. 
# The library DeepXDE is used for all implementation below.
#################################################################################

import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
import deepxde as dde
import os
import shutil
from pathlib import Path
import time

hidden_layers = 8
hidden_units = 20
number_of_epochs = 10000
learning_rate = 1e-3

domain_points = 10000
boundary_points = 50
initial_points = 50
test_points = 100000
test_points_x = 256
test_points_t = 101

x_min = -1
x_max = 1
t_min = 0
t_max = 1

nus = [0.0031, 0.0063, 0.0095, 0.0127, 0.0159, 0.019, 0.0222, 0.0254, 0.0286, 0.0318]
epochs_source = np.loadtxt("Neural_Networks/epochs.csv", dtype = 'int32', delimiter = ',')


def visualize_solution(nu):
  # Build pseudo model
  def pde(x, u):
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_t = dde.grad.jacobian(u, x, i=0, j=1)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    return u_t + u * u_x - nu * u_xx

  spatial_domain = dde.geometry.Interval(x_min, x_max)
  temporal_domain = dde.geometry.TimeDomain(t_min, t_max)
  spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

  boundary_condition = dde.DirichletBC(spatio_temporal_domain, lambda x: 0, lambda _, on_boundary: on_boundary)
  initial_condition = dde.IC(spatio_temporal_domain, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)


  data = dde.data.TimePDE(spatio_temporal_domain, pde, [boundary_condition, initial_condition],
                          num_domain=domain_points, num_boundary=boundary_points, num_initial=initial_points, num_test=test_points)

  net = dde.maps.FNN([2] + hidden_layers * [hidden_units] + [1], "tanh", "Glorot normal")

  model = dde.Model(data, net)

  # Load reference solution
  file_name = 'Reference_Solutions/u_exact_nu_{}.mat'.format(nus[i])
  data = scipy.io.loadmat(file_name)
  u_exact = data['usol'].T
  x_test, t_test = np.meshgrid(
      np.linspace(x_min, x_max, test_points_x),
      np.linspace(t_min, t_max, test_points_t)
  )
  X = np.vstack((np.ravel(x_test), np.ravel(t_test))).T

  # Reload model and make predictions
  model_name = 'Neural_Networks/nu_{}/Burger_Equation_Source_Model_nu_{}-{}'.format(nus[i], nus[i], epochs_source[i])
  model.compile("adam", lr = learning_rate)
  model.train(model_restore_path = model_name, epochs = 0)

  u_pred = model.predict(X).reshape(test_points_t, test_points_x)
  f = model.predict(X, operator=pde)

  ax = plt.axes(projection="3d")
  ax.plot_wireframe(x_test, t_test, u_pred)
  ax.set_xlabel('location x')
  ax.set_ylabel('time t')
  ax.set_zlabel('u')
  plt.tight_layout()
  plt.savefig('Predictions/Predicted_solution_nu_{}.png'.format(nus[i]), dpi = 300)

  return 


### Main file ###
# Vizualizations
for i in range(len(nus)):
  dde.apply(visualize_solution, (nus[i],))