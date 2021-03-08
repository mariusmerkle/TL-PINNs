#################################################################################
# This script exploits a database of phyiscs-informed neural networks (PINNs).
# For the one-dimensional Burger equation, several target models are initialized
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

hidden_layers = 8
hidden_units = 20
number_of_epochs = 10000
learning_rate = 1e-3
number_of_strategies = 3

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

nus_source = [0.0031, 0.0063, 0.0095, 0.0127, 0.0159, 0.019, 0.0222, 0.0254, 0.0286, 0.0318]
nus_target = [0.0127, 0.0187, 0.0037, 0.0059, 0.0259, 0.0149, 0.0144, 0.0255, 0.006, 0.0258]
epochs_source = np.loadtxt("../Source/Neural_Networks/epochs.csv", dtype = 'int32', delimiter = ',')

def train_target_model(nus_source, epochs_source, nu_target, optimization_strategy):
  individual_similarities = np.zeros(len(nus_source))
  for k in range(len(nus_source)):
    individual_similarities[k] = np.linalg.norm(nu_target - nus_source[k])/np.linalg.norm(nu_target)

  index = np.unravel_index(np.argmin(individual_similarities, axis=None), individual_similarities.shape)[0]
  similarity = individual_similarities[index]
  closest_nu = nus_source[index]

  file_name =  'Reference_Solutions/u_exact_nu_{}.mat'.format(nu_target)
  data = scipy.io.loadmat(file_name)
  u_exact = data['usol'].T
  x_test, t_test = np.meshgrid(
      np.linspace(x_min, x_max, test_points_x),
      np.linspace(t_min, t_max, test_points_t)
  )
  X = np.vstack((np.ravel(x_test), np.ravel(t_test))).T
  

  def pde(x, u):
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_t = dde.grad.jacobian(u, x, i=0, j=1)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    return u_t + u * u_x - nu_target * u_xx

  spatial_domain = dde.geometry.Interval(x_min, x_max)
  temporal_domain = dde.geometry.TimeDomain(t_min, t_max)
  spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

  boundary_condition = dde.DirichletBC(spatio_temporal_domain, lambda x: 0, lambda _, on_boundary: on_boundary)
  initial_condition = dde.IC(spatio_temporal_domain, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)


  data = dde.data.TimePDE(spatio_temporal_domain, pde, [boundary_condition, initial_condition],
                          num_domain=domain_points, num_boundary=boundary_points, num_initial=initial_points, num_test=test_points)

  net = dde.maps.FNN([2] + hidden_layers * [hidden_units] + [1], "tanh", "Glorot normal")
  model = dde.Model(data, net)


  start = time.time()
  if optimization_strategy == 0:
    print("L-BFGS (random)")

    model.compile("L-BFGS-B")
    model.train()

  elif optimization_strategy == 1:
    print("L-BFGS (smart)")
    model_name_source = '../Source/Neural_Networks/nu_{}/Burger_Equation_Source_Model_nu_{}-{}'.format(closest_nu, closest_nu, epochs_source[index])
    
    model.compile("L-BFGS-B")
    model.train(model_restore_path=model_name_source)

  else:
    print("Adam + L-BFGS (random)")
    
    model.compile("adam", lr = learning_rate)
    model.train(epochs = number_of_epochs)
    model.compile("L-BFGS-B")
    model.train()

  end = time.time()
  length = end - start

  u_pred = model.predict(X).reshape(test_points_t, test_points_x)
  f = model.predict(X, operator=pde)
  residual = np.mean(np.absolute(f))
  l2_difference = dde.metrics.l2_relative_error(u_exact, u_pred)

  return l2_difference, residual, similarity, length


### Main File ### 
# Initialisation
l2_differences = np.zeros((len(nus_target), number_of_strategies))
residuals = np.zeros((len(nus_target), number_of_strategies))
similarities = np.zeros((len(nus_target), number_of_strategies))
times = np.zeros((len(nus_target), number_of_strategies))

# Foder Structure
directory_1 = Path('Neural_Networks')
directory_2 = Path('Results')

if directory_1.exists() and directory_1.is_dir():
  shutil.rmtree(directory_1)

if directory_2.exists() and directory_2.is_dir():
  shutil.rmtree(directory_2)

os.mkdir(directory_1)
os.mkdir(directory_2)

# Network Training
for i in range(len(nus_target)):
  for j in range(number_of_strategies):
    l2_differences[i, j], residuals[i, j], similarities[i, j], times[i, j] = dde.apply(train_target_model, (nus_source, epochs_source, nus_target[i], j))

# Print Information
print("Residuals: ", residuals)
print("L2 differences: ", l2_differences)
print("Similarities: ", similarities)
print("Times: ", times)


# Tables
np.savetxt("Results/residuals.csv", residuals, delimiter=",")
np.savetxt("Results/l2_differences.csv", l2_differences, delimiter=",")
np.savetxt("Results/similarities.csv", similarities, delimiter=",")
np.savetxt("Results/times.csv", times, delimiter=",")