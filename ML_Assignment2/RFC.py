import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

Attributes = pd.read_csv("hm_hospitales_covid_structured_30d_train.csv", na_values=0, na_filter=True)
Outcomes = pd.read_csv("split_train_export_30d.csv")

Data = Attributes
Data = Data.drop(labels=['PATIENT ID', 'admission_datetime'], axis='columns')
Data.loc[Data['sex'] == 'FEMALE', 'sex'] = 0
Data.loc[Data['sex'] == 'MALE', 'sex'] = 1
Data.loc[Data['ed_diagnosis'] == 'sx_breathing_difficulty', 'ed_diagnosis'] = 1
Data.loc[Data['ed_diagnosis'] == 'sx_others', 'ed_diagnosis'] = 2
Data.loc[Data['ed_diagnosis'] == 'sx_flu', 'ed_diagnosis'] = 3
Data.loc[Data['ed_diagnosis'] == 'sx_fever', 'ed_diagnosis'] = 4
Data.loc[Data['ed_diagnosis'] == 'sx_cough', 'ed_diagnosis'] = 5

Data = Data.fillna(Data.mode().iloc[0])
X = Data

#X = X.to_numpy()
Y = Outcomes.drop(labels='PATIENT ID', axis='columns')
Y = Y.to_numpy()
Y = Y.ravel()



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

clf = RandomForestClassifier(min_samples_leaf= 7, n_estimators= 300)
Y_pred_prob = clf.fit(X_train, Y_train).predict_log_proba(X_test)
Y_pred = clf.predict(X_test)
for i in range(0, Y_pred.shape[0]):
        if Y_pred_prob[i][1] > -1.1:
                Y_pred[i] = 1
        else:
                Y_pred[i] = 0
        #print(Y_pred_prob[i][0], Y_pred_prob[i][1], Y_pred[i], Y_test[i])
TN, FN, TP, FP = 0, 0, 0, 0
for i in range(len(Y_test)):
        if Y_test[i] == 0 and Y_pred[i] == 0:
                TN += 1
        if Y_test[i] == 1 and Y_pred[i] == 0:
                FN += 1
        if Y_test[i] == 0 and Y_pred[i] == 1:
                FP += 1
        if Y_test[i] == 1 and Y_pred[i] == 1:
                TP += 1

print("TN:", TN, ", FN:", FN, ", TP:", TP, ", FP:", FP)
precision, recall = (TP/(FP+TP)), (TP/(FN+TP))
print("precision:", precision, ", recall:", recall)
print('F1:',  2 * ((precision*recall)/(precision+recall)))


Attributes = pd.read_csv("fixed_test.csv", na_values=0, na_filter=True)
Output_format = {'PATIENT ID': Attributes['PATIENT ID']}
Output = pd.DataFrame(Output_format)

Data = Attributes
Data = Data.drop(labels=['PATIENT ID', 'admission_datetime'], axis='columns')
Data.loc[Data['sex'] == 'FEMALE', 'sex'] = 0
Data.loc[Data['sex'] == 'MALE', 'sex'] = 1
Data.loc[Data['ed_diagnosis'] == 'sx_breathing_difficulty', 'ed_diagnosis'] = 1
Data.loc[Data['ed_diagnosis'] == 'sx_others', 'ed_diagnosis'] = 2
Data.loc[Data['ed_diagnosis'] == 'sx_flu', 'ed_diagnosis'] = 3
Data.loc[Data['ed_diagnosis'] == 'sx_fever', 'ed_diagnosis'] = 4
Data.loc[Data['ed_diagnosis'] == 'sx_cough', 'ed_diagnosis'] = 5

Data = Data.fillna(Data.mode().iloc[0])
X = Data

Y_pred_prob = clf.predict_log_proba(X)
Y_pred = clf.predict(X)
for i in range(0, Y_pred.shape[0]):
        if Y_pred_prob[i][1] > -1.1:
                Y_pred[i] = 1
        else:
                Y_pred[i] = 0
Output['hospital_outcome'] = Y_pred

#print(Output)
Output.to_csv('107062338.csv', index=False) #output prediction
