import numpy as np

a = np.array([1, 0, 1, 1, 0])
b = np.array([0, 1, 0, 0, 0])

print(a, b)
print((a-b)**2)

sum = 0
for i in (a-b)**2:
    sum+=i

print(sum)