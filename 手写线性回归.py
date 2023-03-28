
import numpy as np
X = np.random.normal(10, 1, 100)
X = np.reshape(X, (20, 5))
noise = np.random.normal(0,1,20)
y = np.dot(X, [6,4,5,3,2]) + 2 +noise
y = np.reshape(y, (20,1))
print(y)


def myLinearRegression(X, y, alpha = 0.001,learning_count = 10000):
    # alpha = 0.001
    # learning_count = 10000
    W = np.zeros(shape=[1,X.shape[1]])
    b = 0

    for i in range(learning_count):
        y_h = np.dot(X, W.T) + b
        lost = y_h - y
        W = W - alpha * (1/len(X)) * np.dot(lost.T, X)
        b = b - alpha * (1/len(X)) * lost.sum()
        cost = (lost**2).sum(axis=0)/(2*len(X))
    #     if i % 1000 ==0:
    #         print(cost)
    return W,b,cost

W,b,cost = myLinearRegression(X,y)
print(f'W={W}\nb={b}\ncost={cost}')


y_pre = np.dot(X, W.T) +b
print(y_pre)


# score = 1 - u/v
u = ((y-y_pre)**2).sum()
v = ((y-y.mean())**2).sum()
score = 1-u/v
print(score)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
score_sk=model.score(X,y)
print(score_sk)

