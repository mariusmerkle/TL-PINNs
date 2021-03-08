#################################################################################
# This script generates a database of phyiscs-informed neural networks (PINNs).
# For the two-dimensional Kovasznay flow, several source models are trained
# and subsequently stored in the folder "Neural Networks". On top, the network's
# predictions with respect to the exact solution are saved in the folder
# "Predictions" and statistics on accuracy and training times are stored in the
# foler "Results". There, residuals corresponds to the loss value of the metric
# employed to minimize the differential equation and l2_difference is the relative
# difference with respect to the exact solution.
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

domain_points = 2601
boundary_points = 400
test_points = 100000

x_min = -0.5
x_max = 1
y_min = -0.5
y_max = 1.5

Res = [20, 40, 60, 80, 100]


def train_source_model(Re):
  path = Path('Neural_Networks', 'Re_{}'.format(Re))

  if path.exists() and path.is_dir():
    shutil.rmtree(path)

  os.mkdir(path)
  
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

  model_name = 'Neural_Networks/Re_{}/Kovasznay_Flow_Source_Model_Re_{}'.format(Re, Re)

  start = time.time()

  model.compile("adam", lr=learning_rate)
  model.train(epochs=number_of_epochs)
  model.compile("L-BFGS-B")
  losshistory, train_state = model.train(model_save_path=model_name)

  end = time.time()
  length = end - start
  
  X = spatial_domain.random_points(test_points)
  output = model.predict(X)
  u_pred = output[:, 0]
  v_pred = output[:, 1]
  p_pred = output[:, 2]

  u_exact = u_func(X).reshape(-1)
  v_exact = v_func(X).reshape(-1)
  p_exact = p_func(X).reshape(-1)

  f = model.predict(X, operator=pde)


  l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
  l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
  l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)
  residual = np.mean(np.absolute(f))

  final_epochs = train_state.epoch

  return l2_difference_u, l2_difference_v, l2_difference_p, residual, length, final_epochs


### Main file ###
# Initialisation
l2_differences_u = np.zeros(len(Res))
l2_differences_v = np.zeros(len(Res))
l2_differences_p = np.zeros(len(Res))
residuals = np.zeros(len(Res))
times = np.zeros(len(Res))
epochs = np.zeros(len(nus))

# Folder Structure
directory_1 = Path('Neural_Networks')
directory_2 = Path('Results')

if directory_1.exists() and directory_1.is_dir():
  shutil.rmtree(directory_1)

if directory_2.exists() and directory_2.is_dir():
  shutil.rmtree(directory_2)

os.mkdir(directory_1)
os.mkdir(directory_2)

# Network Training
for i in range(len(Res)):
  l2_differences_u[i], l2_differences_v[i], l2_differences_p[i], residuals[i], times[i], epochs[i] = dde.apply(train_source_model, (Res[i],))

# Print Information
print("Residuals: ", residuals)
print("Relative L2 Difference u: ", l2_differences_u)
print("Relative L2 Difference v: ", l2_differences_v)
print("Relative L2 Difference p: ", l2_differences_p)
print("Times: ", times)

# Tables
np.savetxt("Results/residuals.csv", residuals, delimiter=",")
np.savetxt("Results/l2_differences_u.csv", l2_differences_u, delimiter=",")
np.savetxt("Results/l2_differences_v.csv", l2_differences_v, delimiter=",")
np.savetxt("Results/l2_differences_p.csv", l2_differences_p, delimiter=",")
np.savetxt("Results/times.csv", times, delimiter=",")
np.savetxt("Neural_Networks/epochs.csv", epochs, delimiter=",")