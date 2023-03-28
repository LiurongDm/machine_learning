import pandas_datareader.data as web
import datetime
start = datetime.datetime(2000,1,1)
end = datetime.datetime(2021,8,1)
df = web.DataReader('GOOGL', 'stooq',start, end)
df.sort_index(inplace=True, ascending=True)
df.dropna(inplace=True)
print(df)

predict_count = int(len(df)*0.02)
predict_count = 10
print(predict_count)
# 预测多少数据， 就将实际得收盘价格跟实际的放在同一天，假设预测5天，2021-07-24 预测的就是 2021-07-29 那天的收盘价格
df['label'] = df['Close'].shift(-predict_count)
print(df.head(20))
print(df.tail(20))

X = df.drop(['label'],axis=1)
print(X)
y = df['label'][:-predict_count]

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
print(X)
scale.fit(X)
X = scale.transform(X)
print(X)

X_lately = X[-predict_count:]
X = X[:-predict_count]
print(len(X))
print(len(X_lately))
print(X_lately)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(x_train))
print(len(y_train))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
# model.score(x_train, y_train)
model.score(x_test, y_test)

predict = model.predict(X_lately)
print(len(predict))
print(predict)
import numpy as np
df['predict'] = np.nan
print(len(df))
print(df.tail(20))

import datetime
# print(df.index[-1])
last_date_st = df.index[-1].timestamp()
next_date_st = last_date_st + 86400

# print(next_date)

for i in predict:
    next_date = datetime.datetime.fromtimestamp(next_date_st)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    next_date_st +=  86400

print(df.tail(40))

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')
df['Close'].plot()
df['predict'].plot()
plt.show()