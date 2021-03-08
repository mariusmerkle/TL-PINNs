#################################################################################
# This script evaluates the performance of all target models, i.e. the
# phyiscs-informed neural networks (PINNs) for Burger's equation. 
# The library DeepXDE is used for all implementation below.
#################################################################################

import numpy as np
import matplotlib.pyplot as plt 

residuals = np.loadtxt("Results/residuals.csv", delimiter=",")
l2_differences = np.loadtxt("Results/l2_differences.csv", delimiter=",")
similarities = np.loadtxt("Results/similarities.csv", delimiter=",")
times = np.loadtxt("Results/times.csv", delimiter=",")

similarities = np.ones(np.shape(similarities)) - similarities

indices = np.argsort(similarities, axis = 0)
for i in range(3):
    residuals[:, i] = residuals[indices[:, i], i]
    l2_differences[:, i] = l2_differences[indices[:, i], i]
    similarities[:, i] = similarities[indices[:, i], i]
    times[:, i] = times[indices[:, i], i]


average_residual = np.mean(residuals, axis = 0)
average_l2_difference = np.mean(l2_differences, axis = 0)
average_similarity = np.mean(similarities)
average_time = np.mean(times, axis = 0)


#print("\nResiduals: ", residuals)
#print("\nL2 Differences: ", l2_differences)
#print("\nSimilarities: ", similarities)
#print("\nTimes: ", times)


print("\nAverage Residual: ", average_residual)
print("\nAverage L2 Difference: ", average_l2_difference)
print("\nAverage Similarity: ", average_similarity)
print("\nAverage Time: ", average_time)

### Metrics
average_speed_up = times[:, 2]/times[:, 1]
accuracy_comparison = np.mean(l2_differences[:, 2]/l2_differences[:, 1])
print(average_speed_up)
print(accuracy_comparison)


### Figures
# Times: L-BFGS (smart) over similarity
plt.figure(0)
plt.plot(similarities[:, 1], times[:, 1], color='red')
plt.xlabel(r'Target Model')
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