#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
single_node=[]
with open(sys.argv[1]) as f:
    for line in f:
        single_node.append(float(line.strip('\n').split()[0]))
f.close()
plt.plot(single_node,'g:')
plt.show()
