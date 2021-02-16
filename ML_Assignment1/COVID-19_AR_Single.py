import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt
# 自訂MAPE計算公式
def MAPE(pred, test):
  pred, test = np.array(pred), np.array(test)
  sum = 0
  for i in range(7):
    if test[i] != 0:
      sum += abs((test[i] - pred[i]) / test[i])
  return (sum / 7) * 100

#匯入data，取出國家
Data = pd.read_excel("COVID-19-10-08.xlsx")
Country_List = Data['countriesAndTerritories'].unique()
Output = pd.DataFrame(columns = Country_List)

##取出該國家資料
df = Data.loc[Data['countriesAndTerritories'] == 'South_Korea']

data = df['cases'].values #取出每日cases數
data = data[::-1] #first => data[0], last => data[len(data)-1]

cases = df['cases'].values
cases = cases[::-1]
date = np.arange(1, len(cases) + 1)
data = np.column_stack((date, cases))
data = pd.DataFrame(data=data, columns=['date', 'cases'])

clf=IsolationForest()
label = clf.fit_predict(data)
data['outlier'] = label

dispose_data = data.loc[data['outlier'] == 1]

mean = int(np.mean(data['cases']))

data.loc[data['outlier'] == -1, 'cases'] = mean
replace_data = data

data = replace_data['cases'].values

#cases負數修正
for x in np.nditer(data, op_flags=['readwrite']):
    x[...] = int(x)
    if x < 0:
        x[...] = abs(x)

date = np.arange(1, len(data) + 1) #first => date[0], last => date[len(data)-1]

X, Y = date, data
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

#設定test
X_test = X[len(X) - 7:]
Y_test = Y[len(Y) - 7:]
Y_selftest = Y[len(Y) - 14:len(Y) - 7]
#設定一個高mape以利低mape做取代，設定default  days
minmape = 9999999
days = 100

##取不同時間長度來做training，取mape小的時間長度
for i in range(2, 21):

    if len(Y) > i * 10:
        length = i * 10
    else:
        length = len(Y)
        
    Y_selftrain = Y[len(Y) - length:len(Y) - 14]

    try:

        model = AR(Y_selftrain)
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(Y_selftrain), end=len(Y_selftrain)+len(Y_selftest)-1, dynamic=False)

        for x in np.nditer(predictions, op_flags=['readwrite']):  #修正
            x[...] = int(x)
            if x < 0:
                x[...] = 0

        mape = MAPE(predictions, Y_selftest)
        if mape < minmape:
            days = length
            minmape = mape
    except:
        print('''=================
                ===Model Error===
                =================''')


X_train = X[len(X) - 200 + 7:len(X) - 7]
Y_train = Y[len(Y) - 200 + 7:len(Y) - 7]

model = AR(Y_train)
model_fit = model.fit()
print(model_fit)
predictions = model_fit.predict(start=len(Y_train), end=len(Y_train)+len(Y_test)-1, dynamic=False) 

for x in np.nditer(predictions, op_flags=['readwrite']):  #修正
    x[...] = int(x)
    if x < 0:
        x[...] = 0

mape = MAPE(predictions, Y_test)

print(Y_test)
print(predictions)
print('mape = ', mape)
print('days = ', days)
plt.plot(X[250:],Y[250:], color='red')
plt.plot(X_test,Y_test, color='blue')
plt.plot(X_test,predictions, color='green')
plt.title('Cases Vs Time(AR) South_Korea', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Cases', fontsize=14)
plt.grid(True)
plt.show()