from numpy import *;
import numpy as np
import matplotlib.pyplot as plt

time = []
tips = []
with open('iota') as f:
    for line in f:
        row = line.split();
        time.append(float(row[3]));
        tips.append(float(row[2]));

plt.figure(figsize=(50,50))
plt.scatter(time, tips)
plt.show()