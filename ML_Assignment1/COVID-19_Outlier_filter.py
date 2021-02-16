import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
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
df = Data.loc[Data['countriesAndTerritories'] == 'Aruba']

cases = df['cases'].values
cases = cases[::-1]
date = np.arange(1, len(cases) + 1)
data = np.column_stack((date, cases))
data = pd.DataFrame(data=data, columns=['date', 'cases'])

clf=IsolationForest(contamination=0.2)
label = clf.fit_predict(data)

data['outlier'] = label

dispose_data = data.loc[data['outlier'] == 1]

print(dispose_data)

mean = int(np.mean(data['cases']))

data.loc[data['outlier'] == -1, 'cases'] = mean
replace_data = data

plt.plot(dispose_data['date'],dispose_data['cases'], color='red')
plt.plot(date,cases, color='blue')
# plt.plot(X_test,Y_test, color='blue')
# plt.plot(X_test,predictions, color='green')
plt.title('Cases Vs Time(Outlier) Aruba', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Cases', fontsize=14)
plt.grid(True)
plt.show()

data = df['cases'].values
data = data[::-1]
date = np.arange(1, len(data) + 1)
X, Y = date, data
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

period_length = 100

X_train = X[len(X) - period_length:len(X) - 7]
Y_train = Y[len(Y) - period_length:len(Y) - 7]
X_test = X[len(X) - 7:]
Y_test = Y[len(Y) - 7:]


# Linear
linearModel = LinearRegression() 
linearModel.fit(np.column_stack((X_train, np.power(X_train, 2), np.power(X_train, 3), np.power(X_train, 4), np.power(X_train, 5))), Y_train) 


pred = linearModel.predict(np.column_stack((X_test, np.power(X_test, 2), np.power(X_test, 3), np.power(X_test, 4), np.power(X_test, 5))))
#(X_test, np.power(X_test, 2), np.power(X_test, 3), np.power(X_test, 4), np.power(X_test, 5))
print(Y_test)
print(pred)
print('mape = ', MAPE(pred, Y_test))
plt.scatter(X_train, Y_train, color='red')
plt.scatter(X_test,Y_test, color='blue')
plt.scatter(X_test,pred, color='green')
plt.title('Cases Vs Time(Outlier)', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Cases', fontsize=14)
plt.grid(True)
plt.show()
