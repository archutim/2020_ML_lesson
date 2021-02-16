import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, neural_network, naive_bayes
from sklearn.linear_model import Perceptron

Attributes = pd.read_csv("hm_hospitales_covid_structured_30d_train.csv", na_values=0, na_filter=True)
Outcomes = pd.read_csv("split_train_export_30d.csv")

Output_format = {'PATIENT ID': Attributes['PATIENT ID'], 'hospital_outcome': np.zeros(1834, dtype=int)}
Output = pd.DataFrame(Output_format)

X = Attributes.drop(labels=['PATIENT ID', 'admission_datetime'], axis='columns')
X.loc[X['sex'] == 'FEMALE', 'sex'] = 0
X.loc[X['sex'] == 'MALE', 'sex'] = 1
X.loc[X['ed_diagnosis'] == 'sx_breathing_difficulty', 'ed_diagnosis'] = 1
X.loc[X['ed_diagnosis'] == 'sx_others', 'ed_diagnosis'] = 2
X.loc[X['ed_diagnosis'] == 'sx_flu', 'ed_diagnosis'] = 3
X.loc[X['ed_diagnosis'] == 'sx_fever', 'ed_diagnosis'] = 3
X.loc[X['ed_diagnosis'] == 'sx_cough', 'ed_diagnosis'] = 3
X = X[['age', 'sex', 'ed_diagnosis', 'pmhx_diabetes', 'pmhx_hld', 'pmhx_htn', 'pmhx_ihd'
        , 'pmhx_copd', 'pmhx_activecancer', 'pmhx_chronicliver', 'pmhx_stroke', 'pmhx_chf', 'pmhx_dementia']]

X = X.fillna(X.median())
X = X.to_numpy()
Y = Outcomes.drop(labels='PATIENT ID', axis='columns')
Y = Y.to_numpy()
Y = Y.ravel()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

GNB = naive_bayes.GaussianNB()

Y_pred = GNB.fit(X_train, Y_train).predict(X_test)

print(Y_pred)

print("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (Y_test != Y_pred).sum()))

count = 0
pred_1 = 0
for i in range(len(Y_test)):
        if Y_test[i] == Y_pred[i]:
                if Y_test[i] == 1:
                        count+=1
        if Y_pred[i] == 1:
                pred_1+=1
print('Dead Hit:', count)
print('Dead Miss:', pred_1 - count)        