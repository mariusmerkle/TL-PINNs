#################################################################################
# This script evaluates the performance of all target models, i.e. the
# phyiscs-informed neural networks (PINNs) for the Kovasznay flow. 
# The library DeepXDE is used for all implementation below.
#################################################################################

import numpy as np
import matplotlib.pyplot as plt 

residuals = np.loadtxt("Results/residuals.csv", delimiter=",")
l2_differences_u = np.loadtxt("Results/l2_differences_u.csv", delimiter=",")
l2_differences_v = np.loadtxt("Results/l2_differences_v.csv", delimiter=",")
l2_differences_p = np.loadtxt("Results/l2_differences_p.csv", delimiter=",")
similarities = np.loadtxt("Results/similarities.csv", delimiter=",")
times = np.loadtxt("Results/times.csv", delimiter=",")

#print(similarities)
indices = np.argsort(similarities, axis = 0)
for i in range(3):
    residuals[:, i] = residuals[indices[:, i], i]
    l2_differences_u[:, i] = l2_differences_u[indices[:, i], i]
    l2_differences_v[:, i] = l2_differences_v[indices[:, i], i]
    l2_differences_p[:, i] = l2_differences_p[indices[:, i], i]
    similarities[:, i] = similarities[indices[:, i], i]
    times[:, i] = times[indices[:, i], i]


average_residual = np.mean(residuals, axis = 0)
average_l2_difference_u = np.mean(l2_differences_u, axis = 0)
average_l2_difference_v = np.mean(l2_differences_v, axis = 0)
average_l2_difference_p = np.mean(l2_differences_p, axis = 0)
average_similarity = np.mean(similarities)
average_time = np.mean(times, axis = 0)


#print("\nResiduals: ", residuals)
#print("\nL2 Differences u: ", l2_differences_u)
#print("\nL2 Differences v: ", l2_differences_v)
#print("\nL2 Differences p: ", l2_differences_p)
#print("\nSimilarities: ", similarities)
print("\nTimes: ", times)


print("\nAverage Residual: ", average_residual)
print("\nAverage L2 Difference u: ", average_l2_difference_u)
print("\nAverage L2 Difference v: ", average_l2_difference_v)
print("\nAverage L2 Difference p: ", average_l2_difference_p)
#print("\nAverage Similarity: ", average_similarity)
print("\nAverage Time: ", average_time)

### Metrics
reference = 2 # 0 for L-BFGS (random), 2 for Adam + L-BFGS (random)
speed_up = times[:, reference]/times[:, 1]
accuracy_comparison_u = l2_differences_u[:, reference]/l2_differences_u[:, 1]
accuracy_comparison_v = l2_differences_v[:, reference]/l2_differences_v[:, 1]
accuracy_comparison_p = l2_differences_p[:, reference]/l2_differences_p[:, 1]
accuracies = [accuracy_comparison_u, accuracy_comparison_v, accuracy_comparison_p]
accuracy = np.mean(accuracies, axis = 0)
print("Gain in speed: ", speed_up)
print("Gain in accuracy: ", accuracy)


### Figures
# Times: L-BFGS (random) over similarity
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