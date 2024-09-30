"""Use MontCarlo to test if the linear combination of Gaussion distributions with 0 mean and different variances is a normal Gaussion distribution

https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
"""

# creat two Gaussion distributions
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

g1 = np.random.normal(0, 1, 1000000)
g2 = np.random.normal(0, 2, 1000000)

std_dev1 = 1
std_dev2 = 2

z1, p1 = stats.kstest(g1, 'norm', args=(0, std_dev1))
print('z1:', z1, 'p1:', p1)

z2, p2 = stats.kstest(g2, 'norm', args=(0, std_dev2))
print('z2:', z2, 'p2:', p2)

g_sum = g1 + g2
mean = np.mean(g_sum)
std_dev = np.sqrt(np.var(g_sum))
std_dev3 = np.sqrt(std_dev1**2 + std_dev2**2)

z, p = stats.kstest(g_sum, 'norm', args=(0, std_dev3))
print('z:', z, 'p:', p)
print('mean:', mean, 'std_dev:', std_dev)
if p < 0.05:
    print('Not a normal distribution')

g3 = np.random.normal(0, std_dev3, 1000000)
# visualize them
plt.hist(g_sum, bins=100, alpha=0.5, density=True, label='g_sum')
plt.hist(g3, bins=100, alpha=0.5, density=True, label='g3')
plt.legend()
plt.show()
plt.savefig('GMM.png')
