import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima

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

#Build各個國家的ARIMA model並進行預測
for country in Country_List:
    print('===================\n===================\n', country, '\n===================\n===================')
    ##取出該國家資料
    df = Data.loc[Data['countriesAndTerritories'] == country]

    data = df['cases'].values #取出每日cases數
    data = data[::-1] #first => data[0], last => data[len(data)-1]

    #cases負數修正
    for x in np.nditer(data, op_flags=['readwrite']):
      x[...] = int(x)
      if x < 0:
        x[...] = abs(x)

    #設定test data以及selftest data
    Y = data
    Y = Y.reshape(-1, 1)
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

        model = auto_arima(Y_selftrain, start_p=0, start_q=0, start_P=0, start_Q=0, max_p=100, max_d=100, max_q=100, max_P=100, max_D=100, max_Q=100, m=5)
        predictions = model.predict(n_periods=7)

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

    Y_train = Y[len(Y) - days:]

    try:
      model = auto_arima(Y_train, start_p=0, start_q=0, start_P=0, start_Q=0, max_p=100, max_d=100, max_q=100, max_P=100, max_D=100, max_Q=100, m=5)
    except:
      model = auto_arima(Y_train, start_p=0, start_q=0, start_P=0, start_Q=0, max_p=100, max_d=100, max_q=100, max_P=100, max_D=100, max_Q=100)
    
    predictions = model.predict(n_periods=7) 

    for x in np.nditer(predictions, op_flags=['readwrite']):  #修正
      x[...] = int(x)
      if x < 0:
        x[...] = 0

    #mape = MAPE(predictions, Y_test)
    #predictions = np.append(predictions, mape)
    predictions = np.reshape(predictions, 7)
    Output[country] = predictions
  
Output.to_csv('COVID-19_Predict_ARIMA_Final.csv', index=True) 