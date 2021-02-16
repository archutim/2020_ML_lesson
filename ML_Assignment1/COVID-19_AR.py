import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR

def MAPE(pred, test):
  pred, test = np.array(pred), np.array(test)
  sum = 0
  for i in range(7):
    if test[i] != 0:
      sum += abs((test[i] - pred[i]) / test[i])
  return (sum / 7) * 100

Data = pd.read_excel("COVID-19-10-01.xlsx")
Country_List = Data['countriesAndTerritories'].unique()
Output = pd.DataFrame(columns = Country_List)

for country in Country_List:
    ##取出該國家資料
    df = Data.loc[Data['countriesAndTerritories'] == country]

    data = df['cases'].values #取出每日cases數
    data = data[::-1] #first => data[0], last => data[len(data)-1]

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
    Y_selftest = Y[len(Y) - 7:]
    #設定一個高mape以利低mape做取代，設定default  days
    minmape = 999999
    days = 100

    ##取不同時間長度來做training，取mape小的時間長度
    for i in range(4, 21):

        if len(Y) > i * 10:
            length = i * 10
        else:
            length = len(Y)
            
        Y_selftrain = Y[len(Y) - length:len(Y) - 7]

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
    X_train = X[len(X) - days + 7:len(X) - 7]
    Y_train = Y[len(Y) - days:]

    model = AR(Y_train)
    model_fit = model.fit()

    predictions = model_fit.predict(start=len(Y_train), end=len(Y_train)+7-1, dynamic=False) 

    for x in np.nditer(predictions, op_flags=['readwrite']):  #修正
        x[...] = int(x)
        if x < 0:
            x[...] = 0

#   mape = MAPE(predictions, Y_test)
#   predictions = np.append(predictions, mape)
    predictions = np.reshape(predictions, 7)
    Output[country] = predictions
  
Output.to_csv('COVID-19_Predict_AR.csv', index=True) 