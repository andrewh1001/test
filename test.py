import numpy as np
from pandas import read_csv
datapath = 'poly.txt'
degrees = 1
paramFits = []
'''
data = read_csv(datapath, sep = ' ', header=None)
#data = data['data'].astype(float)
#data = data['data'].str.split(' ', expand = True)
data = np.array(data)
feature_x = data[:,0]
print(data)
print(feature_x)
'''
x = [1,2,3,4]
d = 3
X = [[y**n for n in range(d, -1, -1)] for y in x] 
print(X)