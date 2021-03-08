#################################################################################
# This script generates predictions for all models stored in the database of
# phyiscs-informed neural networks (PINNs) for the Kovasznay flow. 
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

hidden_layers = 4
hidden_units = 50
number_of_epochs = 10000
learning_rate = 1e-3

domain_points = 10000
boundary_points = 50
initial_points = 50
test_points = 100000
test_points_x = 256
test_points_y = 256

x_min = -0.5
x_max = 1
y_min = -0.5
y_max = 1.5

Res_source = [20, 40, 60, 80, 100]
epochs_source = np.loadtxt("Neural_Networks/epochs.csv", dtype = 'int32', delimiter = ',')


def visualize_solution(Re):
  # Build pseudo model
  def pde(x, u):
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:]
    u_vel_x = dde.grad.jacobian(u, x, i = 0, j = 0)
    u_vel_y = dde.grad.jacobian(u, x, i = 0, j = 1)
    u_vel_xx = dde.grad.hessian(u, x, component = 0, i = 0, j = 0)
    u_vel_yy = dde.grad.hessian(u, x, component = 0, i = 1, j = 1)

    v_vel_x = dde.grad.jacobian(u, x, i = 1, j = 0)
    v_vel_y = dde.grad.jacobian(u, x, i = 1, j = 1)
    v_vel_xx = dde.grad.hessian(u, x, component = 1, i = 0, j = 0)
    v_vel_yy = dde.grad.hessian(u, x, component = 1, i = 1, j = 1)

    p_x = dde.grad.jacobian(u, x, i = 2, j = 0)
    p_y = dde.grad.jacobian(u, x, i = 2, j = 1)

    momentum_x = u_vel*u_vel_x + v_vel*u_vel_y + p_x - 1/Re*(u_vel_xx + u_vel_yy)
    momentum_y = u_vel*v_vel_x + v_vel*v_vel_y + p_y - 1/Re*(v_vel_xx + v_vel_yy)
    continuity = u_vel_x + v_vel_y

    return [momentum_x, momentum_y, continuity]


  nu = 1/Re
  l = 1/(2*nu) - np.sqrt(1/(4*nu**2) + 4*np.pi**2)
  def u_func(x):
        return 1 - np.exp(l*x[:, 0:1])*np.cos(2*np.pi*x[:, 1:2])

  def v_func(x):
    return l/(2*np.pi)*np.exp(l*x[:, 0:1])*np.sin(2*np.pi*x[:, 1:2])

  def p_func(x):
    return 1/2*(1 - np.exp(2*l*x[:, 0:1]))



  spatial_domain = dde.geometry.geometry_2d.Rectangle(xmin = [x_min, y_min], xmax = [x_max, y_max])
  def boundary_outflow(x, on_boundary):
    return on_boundary and spatial_domain.on_boundary(x) and np.isclose(x[1], y_max)


  boundary_condition_u = dde.DirichletBC(spatial_domain, u_func, lambda _, on_boundary: on_boundary, component=0)
  boundary_condition_v = dde.DirichletBC(spatial_domain, v_func, lambda _, on_boundary: on_boundary, component=1)
  boundary_condition_right_p = dde.DirichletBC(spatial_domain, p_func, boundary_outflow, component = 2)



  data = dde.data.TimePDE(spatial_domain, pde, [boundary_condition_u, boundary_condition_v, boundary_condition_right_p],
                          num_domain=domain_points, num_boundary=boundary_points, num_test=test_points)

  net = dde.maps.FNN([2] + hidden_layers * [hidden_units] + [3], "tanh", "Glorot normal")

  model = dde.Model(data, net)

  # Compute reference solution
  x_test, y_test = np.meshgrid(
      np.linspace(x_min, x_max, test_points_x),
      np.linspace(y_min, y_max, test_points_y)
  )
  X = np.vstack((np.ravel(x_test), np.ravel(y_test))).T

  # Reload model and make predictions
  model_name = 'Neural_Networks/Re_{}/Kovasznay_Flow_Source_Model_Re_{}-{}'.format(Re, Re, epochs_source[i])
  model.compile("adam", lr = learning_rate)
  model.train(model_restore_path = model_name, epochs = 0)

  output = model.predict(X)
  u_pred = output[:, 0].reshape(test_points_x, test_points_y)
  v_pred = output[:, 1].reshape(test_points_x, test_points_y)
  p_pred = output[:, 2].reshape(test_points_x, test_points_y)

  plt.figure(0)
  ax = plt.axes(projection="3d")
  ax.plot_wireframe(x_test, y_test, u_pred)
  ax.set_xlabel('location x')
  ax.set_ylabel('location y')
  ax.set_zlabel('u')
  plt.tight_layout()
  plt.savefig('Predictions/Predicted_solution_u_Re_{}.png'.format(Re), dpi = 300)

  plt.figure(2)
  ax = plt.axes(projection="3d")
  ax.plot_wireframe(x_test, y_test, v_pred)
  ax.set_xlabel('location x')
  ax.set_ylabel('location y')
  ax.set_zlabel('v')
  plt.tight_layout()
  plt.savefig('Predictions/Predicted_solution_v_Re_{}.png'.format(Re), dpi = 300)

  plt.figure(2)
  ax = plt.axes(projection="3d")
  ax.plot_wireframe(x_test, y_test, p_pred)
  ax.set_xlabel('location x')
  ax.set_ylabel('location y')
  ax.set_zlabel('p')
  plt.tight_layout()
  plt.savefig('Predictions/Predicted_solution_p_Re_{}.png'.format(Re), dpi = 300)

  return 


### Main file ###
# Vizualizations
for i in range(len(Res_source)):
  dde.apply(visualize_solution, (Res_source[i],))