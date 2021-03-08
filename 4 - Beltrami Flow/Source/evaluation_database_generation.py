#################################################################################
# This script evaluates all models stored in the database of
# phyiscs-informed neural networks (PINNs) for the Kovasznay flow. 
# The library DeepXDE is used for all implementation below.
#################################################################################

import numpy as np


residuals_0 = np.loadtxt("Results/residuals_0.csv", residuals_0, delimiter=",")
l2_differences_u_0 = np.loadtxt("Results/l2_differences_u_0.csv", l2_differences_u_0, delimiter=",")
l2_differences_v_0 = np.loadtxt("Results/l2_differences_v_0.csv", l2_differences_v_0, delimiter=",")
l2_differences_w_0 = np.loadtxt("Results/l2_differences_w_0.csv", l2_differences_w_0, delimiter=",")
l2_differences_p_0 = np.loadtxt("Results/l2_differences_p_0.csv", l2_differences_p_0, delimiter=",")
residuals_1 = np.loadtxt("Results/residuals_1.csv", residuals_1, delimiter=",")
l2_differences_u_0 = np.loadtxt("Results/l2_differences_u_1.csv", l2_differences_u_1, delimiter=",")
l2_differences_v_0 = np.loadtxt("Results/l2_differences_v_1.csv", l2_differences_v_1, delimiter=",")
l2_differences_w_0 = np.loadtxt("Results/l2_differences_w_1.csv", l2_differences_w_1, delimiter=",")
l2_differences_p_0 = np.loadtxt("Results/l2_differences_p_1.csv", l2_differences_p_1, delimiter=",")
times = np.loadtxt("Results/times.csv", times, delimiter=",")

print("At the beginning: ")
print("Residuals: ", residuals_0)
print("L2 Differences u: ", l2_differences_u_0)
print("L2 Differences v: ", l2_differences_v_0)
print("L2 Differences w: ", l2_differences_w_0)
print("L2 Differences p: ", l2_differences_p_0)
print("\n")
print("In the end: ")
print("Residuals: ", residuals_1)
print("L2 Differences u: ", l2_differences_u_1)
print("L2 Differences v: ", l2_differences_v_1)
print("L2 Differences w: ", l2_differences_w_1)
print("L2 Differences p: ", l2_differences_p_1)