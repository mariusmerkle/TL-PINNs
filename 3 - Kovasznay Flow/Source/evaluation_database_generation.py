#################################################################################
# This script evaluates all models stored in the database of
# phyiscs-informed neural networks (PINNs) for the Kovasznay flow. 
# The library DeepXDE is used for all implementation below.
#################################################################################

import numpy as np

residuals = np.loadtxt("Results/residuals.csv", delimiter=",")
l2_differences_u = np.loadtxt("Results/l2_differences_u.csv", delimiter=",")
l2_differences_v = np.loadtxt("Results/l2_differences_v.csv", delimiter=",")
l2_differences_p = np.loadtxt("Results/l2_differences_p.csv", delimiter=",")
times = np.loadtxt("Results/times.csv", delimiter=",")
average_residual = np.mean(residuals)
average_l2_difference_u = np.mean(l2_differences_u)
average_l2_difference_v = np.mean(l2_differences_v)
average_l2_difference_p = np.mean(l2_differences_p)
average_time = np.mean(times)


print("\nResiduals: ", residuals)
print("\nL2 Differences u: ", l2_differences_u)
print("\nL2 Differences v: ", l2_differences_v)
print("\nL2 Differences p: ", l2_differences_p)
print("\nTimes: ", times)

print("\nAverage Residual: ", average_residual)
print("\nAverage L2 Difference u: ", average_l2_difference_u)
print("\nAverage L2 Difference v: ", average_l2_difference_v)
print("\nAverage L2 Difference p: ", average_l2_difference_p)
print("\nAverage Time: ", average_time)