#################################################################################
# This script exploits a database of phyiscs-informed neural networks (PINNs).
# For the one-dimensional Poisson equation, several target models are initialized
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
import tensorflow as tf
  

hidden_layers = 3
hidden_units = 20
number_of_epochs = 100
learning_rate = 1e-3

domain_points = 50
boundary_points = 10
test_points = 100

print_every = 10

x_min = -1
x_max = 1

nus_source = [1, 2, 3]
nus_target = [1.4, 2.9]
epochs_source = np.loadtxt("../Source/Neural_Networks/epochs.csv", dtype = 'int32', delimiter = ',')


def train_target_model(nus_source, nu_target, epochs_source, optimization_strategy):
  individual_similarities = np.zeros(len(nus_source))
  for k in range(len(nus_source)):
    individual_similarities[k] = 1 - np.linalg.norm(nu_target - nus_source[k])/np.linalg.norm(nu_target)

  index = np.unravel_index(np.argmax(individual_similarities, axis=None), individual_similarities.shape)[0]
  similarity = individual_similarities[index]
  closest_nu = nus_source[index]
  closest_epochs = epochs_source[index]


  def pde(x, u):
    du_xx = dde.grad.hessian(u, x)
    return du_xx + nu_target * np.pi ** 2 * tf.sin(np.pi * x)
  
  def func(x):
    return nu_target*np.sin(np.pi * x)

  spatial_domain = dde.geometry.Interval(x_min, x_max)

  boundary_condition = dde.DirichletBC(spatial_domain, lambda x: 0, lambda _, on_boundary: on_boundary)


  data = dde.data.PDE(spatial_domain, pde, [boundary_condition],
                          num_domain=domain_points, num_boundary=boundary_points, solution = func, num_test=test_points)

  net = dde.maps.FNN([1] + hidden_layers * [hidden_units] + [1], "tanh", "Glorot normal")

  model = dde.Model(data, net)

  model_name = '../Source/Neural_Networks/nu_{}/Poisson_Equation_Source_Model_nu_{}-{}'.format(closest_nu, closest_nu, closest_epochs)

  start = time.time()
  if optimization_strategy == 0:
    print("Adam (random)")
    movie = dde.callbacks.MovieDumper("Results/movie_random", [x_min], [x_max], period=print_every, y_reference=func)

    model.compile("adam", lr = learning_rate)
    model.train(epochs = number_of_epochs, callbacks = [movie])

  else:
    print("Adam (smart)")
    movie = dde.callbacks.MovieDumper("Results/movie_smart", [x_min], [x_max], period=print_every, y_reference=func)
    
    model.compile("adam", lr = learning_rate)
    model.train(epochs = number_of_epochs, callbacks = [movie], model_restore_path = model_name)


  end = time.time()
  length = end - start

  X = spatial_domain.random_points(test_points)
  u_pred = model.predict(X)
  u_exact = func(X)
  f = model.predict(X, operator=pde)

  residual = np.mean(np.absolute(f))
  l2_difference = dde.metrics.l2_relative_error(u_exact, u_pred)

  return l2_difference, residual, length


### Main file ###
# Folder Structure
directory = Path('Results')

if directory.exists() and directory.is_dir():
  shutil.rmtree(directory)

os.mkdir(directory)

# Network Training
l2_difference, residual, time = train_target_model(nus_source, nus_target[1], epochs_source, optimization_strategy = 1)