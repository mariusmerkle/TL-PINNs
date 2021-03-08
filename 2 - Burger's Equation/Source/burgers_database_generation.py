#################################################################################
# This script generates a database of phyiscs-informed neural networks (PINNs).
# For the one-dimensional Burger equation, several source models are trained
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


def train_source_model(nu):
  path = Path('Neural_Networks', 'nu_{}'.format(nus[i]))

  if path.exists() and path.is_dir():
    shutil.rmtree(path)

  os.mkdir(path)
  
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

  file_name = 'Reference_Solutions/u_exact_nu_{}.mat'.format(nus[i])
  data = scipy.io.loadmat(file_name)
  u_exact = data['usol'].T
  x_test, t_test = np.meshgrid(
      np.linspace(x_min, x_max, test_points_x),
      np.linspace(t_min, t_max, test_points_t)
  )
  X = np.vstack((np.ravel(x_test), np.ravel(t_test))).T

  model_name = 'Neural_Networks/nu_{}/Burger_Equation_Source_Model_nu_{}'.format(nus[i], nus[i])

  start = time.time()

  model.compile("adam", lr=1e-3)
  model.train(epochs=number_of_epochs)
  model.compile("L-BFGS-B")
  losshistory, train_state = model.train(model_save_path=model_name)

  end = time.time()
  length = end - start

  u_pred = model.predict(X).reshape(test_points_t, test_points_x)
  f = model.predict(X, operator=pde)
  residual = np.mean(np.absolute(f))
  l2_difference = dde.metrics.l2_relative_error(u_exact, u_pred)

  final_epochs = train_state.epoch

  return l2_difference, residual, length, final_epochs


### Main file ###
# Initialisation
l2_differences = np.zeros(len(nus))
residuals = np.zeros(len(nus))
times = np.zeros(len(nus))
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
for i in range(len(nus)):
  l2_differences[i], residuals[i], times[i], epochs[i] = dde.apply(train_source_model, (nus[i],))

# Print Information
print("Residuals: ", residuals)
print("L2 differences: ", l2_differences)
print("Times: ", times)

# Tables
np.savetxt("Results/residuals.csv", residuals, delimiter=",")
np.savetxt("Results/l2_differences.csv", l2_differences, delimiter=",")
np.savetxt("Results/times.csv", times, delimiter=",")
np.savetxt("Neural_Networks/epochs.csv", epochs, delimiter=",")