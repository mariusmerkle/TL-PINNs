#################################################################################
# This script generates predictions for all models stored in the database of
# phyiscs-informed neural networks (PINNs) for the Beltrami flow. 
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
number_of_epochs = 20000
learning_rate = 1e-3

domain_points = 50000
boundary_points = 5000
initial_points = 5000
test_points = 1000
test_points_per_dimension = 10

x_min = -1
x_max = 1
y_min = -1
y_max = 1
z_min = -1
z_max = 1
t_min = 0
t_max = 1
z_index = 4

a = 1
Re = 1

ds_source = [0.75, 1]
epochs_source = np.loadtxt("Neural_Networks/epochs.csv", dtype = 'int32', delimiter = ',')

def visualize_solution(d, epochs_source):
  def pde(x, u):
    u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

    u_vel_x = dde.grad.jacobian(u, x, i = 0, j = 0)
    u_vel_y = dde.grad.jacobian(u, x, i = 0, j = 1)
    u_vel_z = dde.grad.jacobian(u, x, i = 0, j = 2)
    u_vel_t = dde.grad.jacobian(u, x, i = 0, j = 3)
    u_vel_xx = dde.grad.hessian(u, x, component = 0, i = 0, j = 0)
    u_vel_yy = dde.grad.hessian(u, x, component = 0, i = 1, j = 1)
    u_vel_zz = dde.grad.hessian(u, x, component = 0, i = 2, j = 2)

    v_vel_x = dde.grad.jacobian(u, x, i = 1, j = 0)
    v_vel_y = dde.grad.jacobian(u, x, i = 1, j = 1)
    v_vel_z = dde.grad.jacobian(u, x, i = 1, j = 2)
    v_vel_t = dde.grad.jacobian(u, x, i = 1, j = 3)
    v_vel_xx = dde.grad.hessian(u, x, component = 1, i = 0, j = 0)
    v_vel_yy = dde.grad.hessian(u, x, component = 1, i = 1, j = 1)
    v_vel_zz = dde.grad.hessian(u, x, component = 1, i = 2, j = 2)

    w_vel_x = dde.grad.jacobian(u, x, i = 2, j = 0)
    w_vel_y = dde.grad.jacobian(u, x, i = 2, j = 1)
    w_vel_z = dde.grad.jacobian(u, x, i = 2, j = 2)
    w_vel_t = dde.grad.jacobian(u, x, i = 2, j = 3)
    w_vel_xx = dde.grad.hessian(u, x, component = 2, i = 0, j = 0)
    w_vel_yy = dde.grad.hessian(u, x, component = 2, i = 1, j = 1)
    w_vel_zz = dde.grad.hessian(u, x, component = 2, i = 2, j = 2)

    p_x = dde.grad.jacobian(u, x, i = 3, j = 0)
    p_y = dde.grad.jacobian(u, x, i = 3, j = 1)
    p_z = dde.grad.jacobian(u, x, i = 3, j = 2)

    momentum_x = u_vel_t + (u_vel*u_vel_x + v_vel*u_vel_y + w_vel*u_vel_z) + p_x - 1/Re*(u_vel_xx + u_vel_yy + u_vel_zz)
    momentum_y = v_vel_t + (u_vel*v_vel_x + v_vel*v_vel_y + w_vel*v_vel_z) + p_y - 1/Re*(v_vel_xx + v_vel_yy + v_vel_zz)
    momentum_z = w_vel_t + (u_vel*w_vel_x + v_vel*w_vel_y + w_vel*w_vel_z) + p_z - 1/Re*(w_vel_xx + w_vel_yy + w_vel_zz)
    continuity = u_vel_x + v_vel_y + w_vel_z

    return [momentum_x, momentum_y, momentum_z, continuity]

  def u_func(x):
    return -a*(np.exp(a*x[:, 0:1])*np.sin(a*x[:, 1:2] + d*x[:, 2:3]) + np.exp(a*x[:, 2:3])*np.cos(a*x[:, 0:1] + d*x[:, 1:2]))*np.exp(-d**2*x[:, 3:4])

  def v_func(x):
    return -a*(np.exp(a*x[:, 1:2])*np.sin(a*x[:, 2:3] + d*x[:, 0:1]) + np.exp(a*x[:, 0:1])*np.cos(a*x[:, 1:2] + d*x[:, 2:3]))*np.exp(-d**2*x[:, 3:4])

  def w_func(x):
    return -a*(np.exp(a*x[:, 2:3])*np.sin(a*x[:, 0:1] + d*x[:, 1:2]) + np.exp(a*x[:, 1:2])*np.cos(a*x[:, 2:3] + d*x[:, 0:1]))*np.exp(-d**2*x[:, 3:4])

  def p_func(x):
    return -1/2*a**2*(np.exp(2*a*x[:, 0:1]) + np.exp(2*a*x[:, 0:1]) + np.exp(2*a*x[:, 2:3]) + 2*np.exp(a*x[:, 0:1] + d*x[:, 1:2])*np.cos(a*x[:, 2:3] + d*x[:, 0:1])*np.exp(a*(x[:, 1:2] + x[:, 2:3])) + 2*np.exp(a*x[:, 1:2] + d*x[:, 2:3])*np.cos(a*x[:, 0:1] + d*x[:, 1:2])*np.exp(a*(x[:, 2:3] + x[:, 0:1])) + 2*np.exp(a*x[:, 2:3] + d*x[:, 0:1])*np.cos(a*x[:, 1:2] + d*x[:, 2:3])*np.exp(a*(x[:, 0:1] + x[:, 1:2])))*np.exp(-2*d**2*x[:, 3:4])

  spatial_domain = dde.geometry.geometry_3d.Cuboid(xmin = [x_min, y_min, z_min], xmax = [x_max, y_max, z_max])
  temporal_domain = dde.geometry.TimeDomain(t_min, t_max)
  spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

  boundary_condition_u = dde.DirichletBC(spatio_temporal_domain, u_func, lambda _, on_boundary: on_boundary, component=0)
  boundary_condition_v = dde.DirichletBC(spatio_temporal_domain, v_func, lambda _, on_boundary: on_boundary, component=1)
  boundary_condition_w = dde.DirichletBC(spatio_temporal_domain, w_func, lambda _, on_boundary: on_boundary, component=2)
  #boundary_condition_p = dde.DirichletBC(spatio_temporal_domain, p_func, lambda _, on_boundary: on_boundary, component=3)

  initial_condition_u = dde.IC(spatio_temporal_domain, u_func, lambda _, on_initial: on_initial, component = 0)
  initial_condition_v = dde.IC(spatio_temporal_domain, v_func, lambda _, on_initial: on_initial, component = 1)
  initial_condition_w = dde.IC(spatio_temporal_domain, w_func, lambda _, on_initial: on_initial, component = 2)
  #initial_condition_p = dde.IC(spatio_temporal_domain, p_func, lambda _, on_initial: on_initial, component = 3)


  data = dde.data.TimePDE(spatio_temporal_domain, pde, [boundary_condition_u, boundary_condition_v, boundary_condition_w,
                                                        initial_condition_u, initial_condition_v, initial_condition_w],
                          num_domain=domain_points, num_boundary=boundary_points, num_initial = initial_points, num_test=test_points)


  net = dde.maps.FNN([4] + hidden_layers * [hidden_units] + [4], "tanh", "Glorot normal")

  model = dde.Model(data, net)

  # Compute reference solution
  x, y, z = np.meshgrid(
      np.linspace(x_min, x_max, test_points_per_dimension),
      np.linspace(y_min, y_max, test_points_per_dimension),
      np.linspace(z_min, z_max, test_points_per_dimension),
  )

  X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T

  t_0 = t_min*np.ones(test_points).reshape(test_points, 1)
  t_1 = t_max*np.ones(test_points).reshape(test_points, 1)

  X_0 = np.hstack((X, t_0))
  X_1 = np.hstack((X, t_1))

  # Reload model and make predictions
  model_name = 'Neural_Networks/d_{}/Beltrami_Flow_Source_Model_d_{}-{}'.format(d, d, epochs_source)
  model.compile("adam", lr = learning_rate)
  model.train(model_restore_path = model_name, epochs = 0)

  output_0 = model.predict(X_0)
  output_1 = model.predict(X_1)

  u_pred_0 = output_0[:, 0].reshape(test_points_per_dimension, test_points_per_dimension, test_points_per_dimension)
  v_pred_0 = output_0[:, 1].reshape(test_points_per_dimension, test_points_per_dimension, test_points_per_dimension)
  w_pred_0 = output_0[:, 2].reshape(test_points_per_dimension, test_points_per_dimension, test_points_per_dimension)
  p_pred_0 = output_0[:, 3].reshape(test_points_per_dimension, test_points_per_dimension, test_points_per_dimension)

  u_pred_1 = output_1[:, 0].reshape(test_points_per_dimension, test_points_per_dimension, test_points_per_dimension)
  v_pred_1 = output_1[:, 1].reshape(test_points_per_dimension, test_points_per_dimension, test_points_per_dimension)
  w_pred_1 = output_1[:, 2].reshape(test_points_per_dimension, test_points_per_dimension, test_points_per_dimension)
  p_pred_1 = output_1[:, 3].reshape(test_points_per_dimension, test_points_per_dimension, test_points_per_dimension)

  plt.figure(0)
  ax = plt.axes(projection="3d")
  ax.plot_wireframe(x[:, :, z_index], y[:, :, z_index], u_pred_0[:, :, z_index])
  ax.set_xlabel('location x')
  ax.set_ylabel('location y')
  ax.set_zlabel('u')
  plt.tight_layout()
  plt.savefig('Predictions/Predicted_solution_u_d_{}_t_0.png'.format(d), dpi = 300)

  plt.figure(1)
  ax = plt.axes(projection="3d")
  ax.plot_wireframe(x[:, :, z_index], y[:, :, z_index], u_pred_1[:, :, z_index])
  ax.set_xlabel('location x')
  ax.set_ylabel('location y')
  ax.set_zlabel('u')
  plt.tight_layout()
  plt.savefig('Predictions/Predicted_solution_u_d_{}_t_1.png'.format(d), dpi = 300)

  plt.figure(2)
  ax = plt.axes(projection="3d")
  ax.plot_wireframe(x[:, :, z_index], y[:, :, z_index], v_pred_0[:, :, z_index])
  ax.set_xlabel('location x')
  ax.set_ylabel('location y')
  ax.set_zlabel('v')
  plt.tight_layout()
  plt.savefig('Predictions/Predicted_solution_v_d_{}_t_0.png'.format(d), dpi = 300)

  plt.figure(3)
  ax = plt.axes(projection="3d")
  ax.plot_wireframe(x[:, :, z_index], y[:, :, z_index], v_pred_1[:, :, z_index])
  ax.set_xlabel('location x')
  ax.set_ylabel('location y')
  ax.set_zlabel('v')
  plt.tight_layout()
  plt.savefig('Predictions/Predicted_solution_v_d_{}_t_1.png'.format(d), dpi = 300)

  plt.figure(4)
  ax = plt.axes(projection="3d")
  ax.plot_wireframe(x[:, :, z_index], y[:, :, z_index], w_pred_0[:, :, z_index])
  ax.set_xlabel('location x')
  ax.set_ylabel('location y')
  ax.set_zlabel('w')
  plt.tight_layout()
  plt.savefig('Predictions/Predicted_solution_w_d_{}_t_0.png'.format(d), dpi = 300)

  plt.figure(5)
  ax = plt.axes(projection="3d")
  ax.plot_wireframe(x[:, :, z_index], y[:, :, z_index], w_pred_1[:, :, z_index])
  ax.set_xlabel('location x')
  ax.set_ylabel('location y')
  ax.set_zlabel('w')
  plt.tight_layout()
  plt.savefig('Predictions/Predicted_solution_w_d_{}_t_1.png'.format(d), dpi = 300)

  plt.figure(6)
  ax = plt.axes(projection="3d")
  ax.plot_wireframe(x[:, :, z_index], y[:, :, z_index], p_pred_0[:, :, z_index])
  ax.set_xlabel('location x')
  ax.set_ylabel('location y')
  ax.set_zlabel('p')
  plt.tight_layout()
  plt.savefig('Predictions/Predicted_solution_p_d_{}_t_0.png'.format(d), dpi = 300)

  plt.figure(7)
  ax = plt.axes(projection="3d")
  ax.plot_wireframe(x[:, :, z_index], y[:, :, z_index], p_pred_1[:, :, z_index])
  ax.set_xlabel('location x')
  ax.set_ylabel('location y')
  ax.set_zlabel('p')
  plt.tight_layout()
  plt.savefig('Predictions/Predicted_solution_p_d_{}_t_1.png'.format(d), dpi = 300)


  return 


### Main file ###
# Vizualizations
for i in range(len(ds_source)):
  dde.apply(visualize_solution, (ds_source[i],epochs_source[i], ))