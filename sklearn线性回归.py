import numpy as np
x = np.random.randint(0,10,10)
print(x)
noise = np.random.normal(0,2,10)
y = x*5+6+noise
print(y)

w = ((x*y).mean()-x.mean()*y.mean())/((x**2).mean()-x.mean()**2)
b = y.mean() - w*x.mean()

y_h = w*x + b
print(y_h)

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x,y_h)
plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = np.reshape(x,(10,1))
model.fit(X, y)
model.score(X,y)


print(model.coef_)
print(model.intercept_)


model.predict(X)