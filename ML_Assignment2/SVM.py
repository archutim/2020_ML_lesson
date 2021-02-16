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
#X = X[['age', 'sex', 'pmhx_diabetes', 'pmhx_ihd', 'lab_sodium', 'lab_alt']]
#X = X[['age', 'lab_sodium', 'lab_urea', 'lab_alt']]
X = X[['age', 'sex', 'ed_diagnosis', 'pmhx_dementia', 'lab_ddimer'
        , 'lab_crp', 'lab_lymphocyte_percentage', 'lab_urea', 'lab_lymphocyte', 'lab_neutrophil_percentage']]
X = X.fillna(X.median())
X = X.to_numpy()
Y = Outcomes.drop(labels='PATIENT ID', axis='columns')
Y = Y.to_numpy()
Y = Y.ravel()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

clf = svm.SVC(kernel='rbf')

clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

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