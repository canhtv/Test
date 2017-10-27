import os
import numpy as np
import pandas as pd

import sklearn.linear_models.LinearRegression
import random
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,10

x = np.array([i * np.pi/180 for i in range(60,300,4)])
np.random.seed(10)
y = np.sin(x) + np.random.normal(0,0.15,len(x))


data = pd.DataFrame(np.column_stack([x,y]),columns = ['x','y'])

for i in range(2,16):
    colname = 'x_%d'%i
    data[colname] = x**i
#print data.head()
#plt.plot(data['x'],data['y'],'.')
#plt.show()

features = ['x', 'x_%d'%i for i in range(2,16)]
X = data[features].values

model = LinearRegression()
model.fit(X,y)

out = model.predict(X)

print sklearn.metrics.mean_squared_error(y,out)
