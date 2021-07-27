#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

def weight_cal(vs, gamma=1.5):
    x_gamma = gamma/2
    
    tmp = (1-vs)
    vc3 = (tmp*vs*vs)*x_gamma
    vc0 = (tmp*tmp*vs)*x_gamma
    vc1 = tmp + 2*vc0 - 1*vc3
    vc2 = vs + 2*vc3 - 1*vc0
    vc0 = (-1)*vc0
    vc3 = (-1)*vc3

    return [vc0, vc1, vc2, vc3]

if len(sys.argv) != 3:
    exit(-1)

res = np.zeros((400))
res_b = np.zeros((400))

t = np.arange(-2, 2, 0.01)

for i in range(100):
    epi = weight_cal(i/100, int(sys.argv[1]))
    res[0 + i]   = epi[0]
    res[200 - i] = epi[1]
    res[300 - i] = epi[2]
    res[300 + i] = epi[3]
    
    epi = weight_cal(i/100, int(sys.argv[2]))
    res_b[0 + i]   = epi[0]
    res_b[200 - i] = epi[1]
    res_b[300 - i] = epi[2]
    res_b[300 + i] = epi[3]
    
plt.plot(t, res, 'r', t, res_b, 'b')
plt.ylabel('some numbers')
plt.show()