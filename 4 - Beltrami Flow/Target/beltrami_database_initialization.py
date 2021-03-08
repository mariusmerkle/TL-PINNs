#################################################################################
# This script exploits a database of phyiscs-informed neural networks (PINNs).
# For the three-dimensional Beltrami flow,, several target models are initialized
# from the previously generated database. For each target model, two different 
# optimization procedures based on random or smart initialization can be chosen.
# A movie showing the neural network's predictions over time is generated.
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
number_of_epochs = 30000
learning_rate = 1e-3
number_of_strategies = 3

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

a = 1
ds_source = [0.75, 1]
ds_target = [0.99, 0.97, 0.89, 0.77, 0.86]
epochs_source = np.loadtxt("../Source/Neural_Networks/epochs.csv", dtype = 'int32', delimiter = ',')
Re = 1

def train_target_model(ds_source, epochs_source, d_target, optimization_strategy):
  individual_similarities = np.zeros(len(ds_source))
  for k in range(len(ds_source)):
    individual_similarities[k] = 1 - np.linalg.norm(d_target - ds_source[k])/np.linalg.norm(d_target)

  index = np.unravel_index(np.argmax(individual_similarities, axis=None), individual_similarities.shape)[0]
  similarity = individual_similarities[index]
  closest_d = ds_source[index]

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
    return -a*(np.exp(a*x[:, 0:1])*np.sin(a*x[:, 1:2] + d_target*x[:, 2:3]) + np.exp(a*x[:, 2:3])*np.cos(a*x[:, 0:1] + d_target*x[:, 1:2]))*np.exp(-d_target**2*x[:, 3:4])

  def v_func(x):
    return -a*(np.exp(a*x[:, 1:2])*np.sin(a*x[:, 2:3] + d_target*x[:, 0:1]) + np.exp(a*x[:, 0:1])*np.cos(a*x[:, 1:2] + d_target*x[:, 2:3]))*np.exp(-d_target**2*x[:, 3:4])

  def w_func(x):
    return -a*(np.exp(a*x[:, 2:3])*np.sin(a*x[:, 0:1] + d_target*x[:, 1:2]) + np.exp(a*x[:, 1:2])*np.cos(a*x[:, 2:3] + d_target*x[:, 0:1]))*np.exp(-d_target**2*x[:, 3:4])

  def p_func(x):
    return -1/2*a**2*(np.exp(2*a*x[:, 0:1]) + np.exp(2*a*x[:, 0:1]) + np.exp(2*a*x[:, 2:3]) + 2*np.exp(a*x[:, 0:1] + d_target*x[:, 1:2])*np.cos(a*x[:, 2:3] + d_target*x[:, 0:1])*np.exp(a*(x[:, 1:2] + x[:, 2:3])) + 2*np.exp(a*x[:, 1:2] + d_target*x[:, 2:3])*np.cos(a*x[:, 0:1] + d_target*x[:, 1:2])*np.exp(a*(x[:, 2:3] + x[:, 0:1])) + 2*np.exp(a*x[:, 2:3] + d_target*x[:, 0:1])*np.cos(a*x[:, 1:2] + d_target*x[:, 2:3])*np.exp(a*(x[:, 0:1] + x[:, 1:2])))*np.exp(-2*d_target**2*x[:, 3:4])

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

  start = time.time()

  if optimization_strategy == 0:
    print("L-BFGS (random)")

    model.compile("L-BFGS-B", loss_weights= [1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
    model.train()

  elif optimization_strategy == 1:
    print("L-BFGS (smart)")
    model_name_source = '../Source/Neural_Networks/d_{}/Beltrami_Flow_Source_Model_d_{}-{}'.format(closest_d, closest_d, epochs_source[index])
    
    model.compile("L-BFGS-B", loss_weights= [1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
    model.train(model_restore_path=model_name_source)

  else:
    print("Adam + L-BFGS (random)")
    
    model.compile("adam", lr = learning_rate, loss_weights= [1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
    model.train(epochs = number_of_epochs)
    model.compile("L-BFGS-B", loss_weights= [1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
    model.train()

  end = time.time()
  length = end - start

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

  output_0 = model.predict(X_0)
  output_1 = model.predict(X_1)

  u_pred_0 = output_0[:, 0].reshape(-1)
  v_pred_0 = output_0[:, 1].reshape(-1)
  w_pred_0 = output_0[:, 2].reshape(-1)
  p_pred_0 = output_0[:, 3].reshape(-1)

  u_exact_0 = u_func(X_0).reshape(-1)
  v_exact_0 = v_func(X_0).reshape(-1)
  w_exact_0 = w_func(X_0).reshape(-1)
  p_exact_0 = p_func(X_0).reshape(-1)

  u_pred_1 = output_1[:, 0].reshape(-1)
  v_pred_1 = output_1[:, 1].reshape(-1)
  w_pred_1 = output_1[:, 2].reshape(-1)
  p_pred_1 = output_1[:, 3].reshape(-1)

  u_exact_1 = u_func(X_1).reshape(-1)
  v_exact_1 = v_func(X_1).reshape(-1)
  w_exact_1 = w_func(X_1).reshape(-1)
  p_exact_1 = p_func(X_1).reshape(-1)


  f_0 = model.predict(X_0, operator=pde)
  f_1 = model.predict(X_1, operator=pde)


  l2_difference_u_0 = dde.metrics.l2_relative_error(u_exact_0, u_pred_0)
  l2_difference_v_0 = dde.metrics.l2_relative_error(v_exact_0, v_pred_0)
  l2_difference_w_0 = dde.metrics.l2_relative_error(w_exact_0, w_pred_0)
  l2_difference_p_0 = dde.metrics.l2_relative_error(p_exact_0, p_pred_0)
  residual_0 = np.mean(np.absolute(f_0))

  l2_difference_u_1 = dde.metrics.l2_relative_error(u_exact_1, u_pred_1)
  l2_difference_v_1 = dde.metrics.l2_relative_error(v_exact_1, v_pred_1)
  l2_difference_w_1 = dde.metrics.l2_relative_error(w_exact_1, w_pred_1)
  l2_difference_p_1 = dde.metrics.l2_relative_error(p_exact_1, p_pred_1)
  residual_1 = np.mean(np.absolute(f_1))

  return l2_difference_u_0, l2_difference_v_0, l2_difference_w_0, l2_difference_p_0, residual_0, l2_difference_u_1, l2_difference_v_1, l2_difference_w_1, l2_difference_p_1, residual_1, similarity, length

### Main file ###
# Initialisation
l2_differences_u_0 = np.zeros((len(ds_target), number_of_strategies))
l2_differences_v_0 = np.zeros((len(ds_target), number_of_strategies))
l2_differences_w_0 = np.zeros((len(ds_target), number_of_strategies))
l2_differences_p_0 = np.zeros((len(ds_target), number_of_strategies))
residuals_0 = np.zeros((len(ds_target), number_of_strategies))
l2_differences_u_1 = np.zeros((len(ds_target), number_of_strategies))
l2_differences_v_1 = np.zeros((len(ds_target), number_of_strategies))
l2_differences_w_1 = np.zeros((len(ds_target), number_of_strategies))
l2_differences_p_1 = np.zeros((len(ds_target), number_of_strategies))
residuals_1 = np.zeros((len(ds_target), number_of_strategies))
similarities = np.zeros((len(ds_target), number_of_strategies))
times = np.zeros((len(ds_target), number_of_strategies))

# Folder Structure
directory = Path('Results')

if directory.exists() and directory.is_dir():
  shutil.rmtree(directory)

os.mkdir(directory)

# Network Training
for i in range(len(ds_target)):
  for j in range(number_of_strategies):
    l2_differences_u_0[i, j], l2_differences_v_0[i, j], l2_differences_w_0[i, j], l2_differences_p_0[i, j], residuals_0[i, j], l2_differences_u_1[i, j], l2_differences_v_1[i, j], l2_differences_w_1[i, j], l2_differences_p_1[i, j], residuals_1[i, j], similarities[i, j], times[i, j] = dde.apply(train_target_model, (ds_source, epochs_source, ds_target[i], j,))


# Print Information
print("At the beginning: ")
print("Residuals: ", residuals_0)
print("Relative L2 Difference u: ", l2_differences_u_0)
print("Relative L2 Difference v: ", l2_differences_v_0)
print("Relative L2 Difference w: ", l2_differences_w_0)
print("Relative L2 Difference p: ", l2_differences_p_0)
print("\n")
print("In the end: ")
print("Residuals: ", residuals_1)
print("Relative L2 Difference u: ", l2_differences_u_1)
print("Relative L2 Difference v: ", l2_differences_v_1)
print("Relative L2 Difference w: ", l2_differences_w_1)
print("Relative L2 Difference p: ", l2_differences_p_1)
print("\n")
print("Times: ", times)
print("Similarities: ", similarities)


# Tables
np.savetxt("Results/residuals_0.csv", residuals_0, delimiter=",")
np.savetxt("Results/l2_differences_u_0.csv", l2_differences_u_0, delimiter=",")
np.savetxt("Results/l2_differences_v_0.csv", l2_differences_v_0, delimiter=",")
np.savetxt("Results/l2_differences_w_0.csv", l2_differences_w_0, delimiter=",")
np.savetxt("Results/l2_differences_p_0.csv", l2_differences_p_0, delimiter=",")
np.savetxt("Results/residuals_1.csv", residuals_1, delimiter=",")
np.savetxt("Results/l2_differences_u_1.csv", l2_differences_u_1, delimiter=",")
np.savetxt("Results/l2_differences_v_1.csv", l2_differences_v_1, delimiter=",")
np.savetxt("Results/l2_differences_w_1.csv", l2_differences_w_1, delimiter=",")
np.savetxt("Results/l2_differences_p_1.csv", l2_differences_p_1, delimiter=",")
np.savetxt("Results/times.csv", times, delimiter=",")
np.savetxt("Results/similarities.csv", similarities, delimiter=",")