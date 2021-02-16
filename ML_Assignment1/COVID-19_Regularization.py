def MAPE(pred, test):
  pred, test = np.array(pred), np.array(test)
  sum = 0
  for i in range(7):
    if test[i] != 0:
      sum += abs((test[i] - pred[i]) / test[i])
  return (sum / 7) * 100

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge

Data = pd.read_excel("COVID-19-10-06.xlsx")
Country_List = Data['countriesAndTerritories'].unique()
Output = pd.DataFrame(columns = Country_List)

#for country in Country_List:
df = Data.loc[Data['countriesAndTerritories'] == 'Aruba']

data = df['cases'].values
data = data[::-1] #first => data[0], last => data[len(data)-1]
date = np.arange(1, len(data) + 1) #first => date[0], last => date[len(data)-1]
X, Y = date, data
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

period_length = 100

X_train = X[len(X) - period_length:len(X) - 7]
Y_train = Y[len(Y) - period_length:len(Y) - 7]
X_test = X[len(X) - 7:]
Y_test = Y[len(Y) - 7:]
X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9, X_train10 = np.power(X_train, 2), np.power(X_train, 3), np.power(X_train, 4), np.power(X_train, 5), np.power(X_train, 6), np.power(X_train, 7), np.power(X_train, 8), np.power(X_train, 9), np.power(X_train, 10)
X_test2, X_test3, X_test4, X_test5, X_test6, X_test7, X_test8, X_test9, X_test10 = np.power(X_test, 2), np.power(X_test, 3), np.power(X_test, 4), np.power(X_test, 5), np.power(X_test, 6), np.power(X_test, 7), np.power(X_test, 8), np.power(X_test, 9), np.power(X_test, 10)
X_train_set = np.column_stack((X_train, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9, X_train10))
X_test_set = np.column_stack((X_test, X_test2, X_test3, X_test4, X_test5, X_test6, X_test7, X_test8, X_test9, X_test10))
# Linear
linearModel = LinearRegression() 
linearModel.fit(X_train_set, Y_train) 
#Lasso
LassoModel = Lasso()
LassoModel.fit(X_train_set, Y_train)
#Ridge
RidgeModel = Ridge()
RidgeModel.fit(X_train_set, Y_train)

pred = linearModel.predict(X_train_set)

for x in np.nditer(pred, op_flags=['readwrite']):  #修正
    x[...] = int(x)
    if x < 0:
        x[...] = 0

print(Y_test)
print(pred)
print('mape = ', MAPE(pred, Y_train))
plt.scatter(X_train, Y_train, color='red')
#plt.scatter(X_test,Y_test, color='blue')
plt.scatter(X_train,pred, color='green')
plt.title('Cases Vs Time(Linear)', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Cases', fontsize=14)
plt.grid(True)
plt.show()