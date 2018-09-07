import numpy as np
import matplotlib.pyplot as plt


time_steps = 1024
x1 = np.linspace(0, 100, time_steps)
y=np.sin(x1)+1.0
q = 2.0/16.0
y = np.floor(y/q)
print(y.min())
print(y.max())
print(len(y))
print(y)
chunk_size = 700
chunk_start = np.random.randint(low=0, high=time_steps-chunk_size)
chunk = y[chunk_start:chunk_size]
