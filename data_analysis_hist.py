import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
"""
test
Number of digits count: {1: 2483, 2: 8356, 3: 2081, 4: 146, 5: 2}
Classes Count: [0, 5099, 4149, 2882, 2523, 2384, 1977, 2019, 1660, 1595, 1744]

train
Number of digits count: {1: 5137, 2: 18130, 3: 8691, 4: 1434, 5: 9, 6: 1}
Classes Count: [0, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 5045, 4659, 4948]
"""


data = {1: 2483, 2: 8356, 3: 2081, 4: 146, 5: 2}
plt.hist(data.keys(), weights=data.values(), bins=range(7))
plt.xlabel('Number of digits')
plt.ylabel('Count')
plt.title('Histogram: Testing Set, Number of Digits')
plt.show()

data = {1:5099, 2:4149, 3:2882, 4:2523, 5:2384,
        6:1977, 7:2019, 8:1660, 9:1595, 0:1744}
plt.hist(data.keys(), weights=data.values(), bins=range(-1, 12))
plt.xlabel('Digit class')
plt.ylabel('Count')
plt.title('Histogram: Testing Set, Digit class')
plt.show()

data = {1: 5137, 2: 18130, 3: 8691, 4: 1434, 5: 9, 6: 1}
plt.hist(data.keys(), weights=data.values(), bins=range(8))
plt.xlabel('Number of digits')
plt.ylabel('Count')
plt.title('Histogram: Training Set, Number of Digits')
plt.show()

data = {1:13861, 2:10585, 3:8497, 4:7458, 5:6882,
        6:5727, 7:5595, 8:5045, 9:4659, 0:4948}
plt.hist(data.keys(), weights=data.values(), bins=range(-1, 12))
plt.xlabel('Digit class')
plt.ylabel('Count')
plt.title('Histogram: Training Set, Digit class')
plt.show()
