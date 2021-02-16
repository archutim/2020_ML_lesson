def MAPE(pred, test):
    pred, test = np.array(pred), np.array(test)
    sum = 0
    for i in range(7):
        if test[i] != 0:
            sum += abs((test[i] - pred[i]) / test[i])
    return (sum / 7) * 100

def X_generator(X):
    X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10, X_sqrt = X, np.power(X, 2), np.power(X, 3), np.power(X, 4), np.power(X, 5), np.power(X, 6), np.power(X, 7), np.power(X, 8), np.power(X, 9), np.power(X, 10), np.sqrt(X)
    X_set = np.column_stack((X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_sqrt))
    return X
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge

Data = pd.read_excel("COVID-19-10-07.xlsx")
Country_List = Data['countriesAndTerritories'].unique()
Output = pd.DataFrame(columns = Country_List)

df = Data.loc[Data['countriesAndTerritories'] == 'Germany']
data = df['cases'].values
data = data[::-1] #first => data[0], last => data[len(data)-1]
date = np.arange(1, len(data) + 1) #first => date[0], last => date[len(data)-1]
X, Y = date, data
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

X_test = X_generator(X[len(X) - 7:])
Y_test = Y[len(Y) - 7:]
Y_selftest = Y[len(Y) - 14:len(Y) - 7]

minmape = 999999  #設定minmape，用來取最小的mape
days = 50 #設定default天數

for i in range(4, 21): #嘗試取不同長度的時間。ex.近10天, 近20天
    if len(data) > i * 10 + 1:
        length = i * 10 + 1
    else:
        length = len(data)

    X_selftrain = X_generator(X[len(X) - length:len(X) - 14])
    Y_selftrain = Y[len(Y) - length:len(Y) - 14]
    X_selftest = X_generator(X[len(X) - 14:len(X) - 7])

    # Linear
    linearModel = LinearRegression(normalize=True) 
    linearModel.fit(X_selftrain, Y_selftrain) 
    #Lasso
    LassoModel = Lasso(normalize=True)
    LassoModel.fit(X_selftrain, Y_selftrain)
    #Ridge
    RidgeModel = Ridge(normalize=True)
    RidgeModel.fit(X_selftrain, Y_selftrain)

    pred = RidgeModel.predict(X_selftest)  #Predict

    for x in np.nditer(pred, op_flags=['readwrite']):  #修正
        x[...] = int(x)
        if x < 0:
            x[...] = 0

    mape = MAPE(pred, Y_selftest)  #計算mape

    if mape < minmape:
        minmape = mape
        days = length


X_train = X_generator(X[len(X) - 30:len(X) - 7])
Y_train = Y[len(Y) - 30:len(Y) - 7]

# Linear
linearModel = LinearRegression(normalize=True) 
linearModel.fit(X_train, Y_train) 
#Lasso
LassoModel = Lasso(normalize=True)
LassoModel.fit(X_train, Y_train)
#Ridge
RidgeModel = Ridge(normalize=True)
RidgeModel.fit(X_train, Y_train)

pred = RidgeModel.predict(X_test)

for x in np.nditer(pred, op_flags=['readwrite']):  #修正
    x[...] = int(x)
    if x < 0:
        x[...] = 0

mape = MAPE(pred, Y_test)
#pred = np.append(pred, mape)
print(days)
print(mape)
print(minmape)

plt.plot(X[len(X) - 30:len(X) - 7],Y_train, color='red')
plt.plot(X[len(X) - 7:],Y_test, color='blue')
plt.plot(X[len(X) - 7:],pred, color='green')
plt.title('Cases Vs Time(Linear) for Germany', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Cases', fontsize=14)
plt.grid(True)
plt.show()