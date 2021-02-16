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
    return X_set
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge #import linear, norm1, norm2 method

Data = pd.read_excel("COVID-19-10-08.xlsx")
Country_List = Data['countriesAndTerritories'].unique() #build country list
Output = pd.DataFrame(columns = Country_List) #declear output dataframe

for country in Country_List:
    df = Data.loc[Data['countriesAndTerritories'] == country]
    data = df['cases'].values   #chose "cases" as data
    data = data[::-1]
    date = np.arange(1, len(data) + 8)
    X, Y = date[:len(date) - 7], data
    
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    X_test = X_generator(date[len(date) - 7:])
    Y_test = Y[len(Y) - 7:]
    Y_selftest = Y[len(Y) - 7:]

    minmape = 999999  #set minmape
    days = 50 #set default days
  
    for i in range(4, 21): #Try different time interval, pick the best duration for training
        if len(data) > i * 10 + 1:
            length = i * 10 + 1
        else:
            length = len(data)

        X_selftrain = X_generator(X[len(X) - length:len(X) - 7])
        Y_selftrain = Y[len(Y) - length:len(Y) - 7]
        X_selftest = X_generator(X[len(X) - 7:])

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

        for x in np.nditer(pred, op_flags=['readwrite']):  #fix float data and negetive data
            x[...] = int(x)
            if x < 0:
                x[...] = 0

        mape = MAPE(pred, Y_selftest)  #calculates mape

        if mape < minmape:
            minmape = mape
            days = length


    X_train = X_generator(X[len(X) - days + 7:])
    Y_train = Y[len(Y) - days + 7:]

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

    for x in np.nditer(pred, op_flags=['readwrite']):  #fix float data and negetive data
        x[...] = int(x)
        if x < 0:
            x[...] = 0

    pred = np.reshape(pred, 7)
    Output[country] = pred

Output.to_csv('COVID-19_Predict_Final.csv', index=True) #output prediction