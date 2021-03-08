#################################################################################
# This script evaluates the performance of all target models, i.e. the
# phyiscs-informed neural networks (PINNs) for the Beltrami flow. 
# The library DeepXDE is used for all implementation below.
#################################################################################

import numpy as np
import matplotlib.pyplot as plt 

number_of_strategies = 3

residuals_0 = np.loadtxt("Results/residuals_0.csv", delimiter=",")
l2_differences_u_0 = np.loadtxt("Results/l2_differences_u_0.csv", delimiter=",")
l2_differences_v_0 = np.loadtxt("Results/l2_differences_v_0.csv", delimiter=",")
l2_differences_w_0 = np.loadtxt("Results/l2_differences_w_0.csv", delimiter=",")
l2_differences_p_0 = np.loadtxt("Results/l2_differences_p_0.csv", delimiter=",")

residuals_1 = np.loadtxt("Results/residuals_1.csv", delimiter=",")
l2_differences_u_1 = np.loadtxt("Results/l2_differences_u_1.csv", delimiter=",")
l2_differences_v_1 = np.loadtxt("Results/l2_differences_v_1.csv", delimiter=",")
l2_differences_w_1 = np.loadtxt("Results/l2_differences_w_1.csv", delimiter=",")
l2_differences_p_1 = np.loadtxt("Results/l2_differences_p_1.csv", delimiter=",")


similarities = np.loadtxt("Results/similarities.csv", delimiter=",")
times = np.loadtxt("Results/times.csv", delimiter=",")

#print(similarities)
indices = np.argsort(similarities, axis = 0)
for i in range(number_of_strategies):
    residuals_0[:, i] = residuals_0[indices[:, i], i]
    l2_differences_u_0[:, i] = l2_differences_u_0[indices[:, i], i]
    l2_differences_v_0[:, i] = l2_differences_v_0[indices[:, i], i]
    l2_differences_w_0[:, i] = l2_differences_w_0[indices[:, i], i]
    l2_differences_p_0[:, i] = l2_differences_p_0[indices[:, i], i]

    residuals_1[:, i] = residuals_1[indices[:, i], i]
    l2_differences_u_1[:, i] = l2_differences_u_1[indices[:, i], i]
    l2_differences_v_1[:, i] = l2_differences_v_1[indices[:, i], i]
    l2_differences_w_1[:, i] = l2_differences_w_1[indices[:, i], i]
    l2_differences_p_1[:, i] = l2_differences_p_1[indices[:, i], i]

    similarities[:, i] = similarities[indices[:, i], i]
    times[:, i] = times[indices[:, i], i]


averages_times = np.mean(times, axis = 0)
average_l2_difference_u_0 = np.mean(l2_differences_u_0, axis = 0)
average_l2_difference_v_0 = np.mean(l2_differences_v_0, axis = 0)
average_l2_difference_w_0 = np.mean(l2_differences_w_0, axis = 0)
average_l2_difference_u_1 = np.mean(l2_differences_u_1, axis = 0)
average_l2_difference_v_1 = np.mean(l2_differences_v_1, axis = 0)
average_l2_difference_w_1 = np.mean(l2_differences_w_1, axis = 0)

speed_up = times[:, 2]/times[:, 1]
print("Gain in speed: ", speed_up)


print(averages_times)
#print(average_l2_difference_u_0)
#print(average_l2_difference_v_0)
#print(average_l2_difference_w_0)
#print(average_l2_difference_u_1)
#print(average_l2_difference_v_1)
#print(average_l2_difference_w_1)



### Figures
# Times: L-BFGS (random) vs L-BFGS (smart) vs Adam + L-BFGS (random)
plt.figure(0)
plt.plot(similarities[:, 1], times[:, 1], color='red')
plt.xlabel(r'Similarity Measure $\mathcal{S}$')
plt.ylabel(r'Time until convergence in seconds')
plt.tight_layout()
plt.savefig('Figures/Time_over_similarity.png', dpi = 600)

# Bar Chart: Speed-up of L-BFGS (smart) with respect to Adam + L-BFGS (random)
width = 0.25
plt.figure(1)
plt.bar(np.arange(len(times[:, 1])) - width/2, times[:, 1], width=width, color='red', label='L-BFGS (smart)')
plt.bar(np.arange(len(times[:, 2])) + width/2, times[:, 2], width=width, color='blue', label='Adam + L-BFGS (random)')
plt.xlabel('Target Model Index')
plt.ylabel(r'Time until convergence in seconds')
plt.legend(loc="lower left", bbox_to_anchor= (0.0, 1.01), ncol=2, borderaxespad=0, frameon=False)
plt.tight_layout()
plt.savefig('Figures/Speed-up.png', dpi = 600)