#################################################################################
# This script evaluates all models stored in the database of
# phyiscs-informed neural networks (PINNs) for Burger's equation. 
# The library DeepXDE is used for all implementation below.
#################################################################################

import numpy as np

residuals = np.loadtxt("Results/residuals.csv", delimiter=",")
l2_differences = np.loadtxt("Results/l2_differences.csv", delimiter=",")
times = np.loadtxt("Results/times.csv", delimiter=",")
average_residual = np.mean(residuals)
average_l2_difference = np.mean(l2_differences)
average_time = np.mean(times)


print("\nResiduals: ", residuals)
print("\nL2 Differences: ", l2_differences)
print("\nTimes: ", times)

print("\nAverage Residual: ", average_residual)
print("\nAverage L2 Difference: ", average_l2_difference)
print("\nAverage Time: ", average_time)