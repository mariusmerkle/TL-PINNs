#################################################################################
# This script generates a database of phyiscs-informed neural networks (PINNs).
# For the one-dimensional Poisson equation, several source models are trained
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
import tensorflow as tf
  

hidden_layers = 3
hidden_units = 20
number_of_epochs = 10000
learning_rate = 1e-3

domain_points = 50
boundary_points = 10
test_points = 100

x_min = -1
x_max = 1

nus = [1, 2, 3]


def train_source_model(nu):
  path = Path('Neural_Networks', 'nu_{}'.format(nus[i]))

  if path.exists() and path.is_dir():
    shutil.rmtree(path)

  os.mkdir(path)
  
  def pde(x, u):
    du_xx = dde.grad.hessian(u, x)
    return du_xx + nu*np.pi ** 2 * tf.sin(np.pi * x)
  
  def func(x):
    return nu*np.sin(np.pi * x)

  spatial_domain = dde.geometry.Interval(x_min, x_max)

  boundary_condition = dde.DirichletBC(spatial_domain, lambda x: 0, lambda _, on_boundary: on_boundary)


  data = dde.data.PDE(spatial_domain, pde, [boundary_condition],
                          num_domain=domain_points, num_boundary=boundary_points, solution = func, num_test=test_points)

  net = dde.maps.FNN([1] + hidden_layers * [hidden_units] + [1], "tanh", "Glorot normal")

  model = dde.Model(data, net)

  model_name = 'Neural_Networks/nu_{}/Poisson_Equation_Source_Model_nu_{}'.format(nus[i], nus[i])

  start = time.time()

  model.compile("adam", lr=1e-3)
  history, train_state = model.train(epochs=number_of_epochs, model_save_path=model_name)

  end = time.time()
  length = end - start

  X = np.linspace(x_min, x_max, test_points).reshape(-1, 1)
  u_pred = model.predict(X)
  u_exact = func(X)
  f = model.predict(X, operator=pde)

  figure_name = 'Predictions/Predicted_solution_nu_{}'.format(nu)
  plt.plot(X, u_exact, color = 'blue', label = 'exact solution')
  plt.plot(X, u_pred, color = 'red', linestyle ='--', label = 'predicted solution')
  plt.xlabel(r'location $x$')
  plt.ylabel(r'$u$')
  plt.legend(loc="upper left")
  plt.tight_layout()
  plt.savefig(figure_name, dpi = 600)

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
directory_3 = Path('Predictions')

if directory_1.exists() and directory_1.is_dir():
  shutil.rmtree(directory_1)

if directory_2.exists() and directory_2.is_dir():
  shutil.rmtree(directory_2)

if directory_3.exists() and directory_3.is_dir():
  shutil.rmtree(directory_3)

os.mkdir(directory_1)
os.mkdir(directory_2)
os.mkdir(directory_3)

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