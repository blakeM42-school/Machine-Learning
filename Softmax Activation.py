import numpy as np

'''
import math
E = math.e
for output in layer_outputs:
    exp_values.append(E**output)
print(exp_values)

norm_base = sum(exp_values)
norm_values = []
for values in exp_values:
    norm_values.append(values / norm_base)
print(norm_values)
print(sum(norm_values))
'''

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]


exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(exp_values)
print(norm_values)





