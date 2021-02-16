import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
Attributes = pd.read_csv("hm_hospitales_covid_structured_30d_train.csv", na_values=0, na_filter=True)
Outcomes = pd.read_csv("split_train_export_30d.csv")

Output_format = {'PATIENT ID': Attributes['PATIENT ID'], 'hospital_outcome': np.zeros(1834, dtype=int)}
Output = pd.DataFrame(Output_format)

Data = Attributes
Data.loc[Data['sex'] == 'FEMALE', 'sex'] = 0
Data.loc[Data['sex'] == 'MALE', 'sex'] = 1
Data.loc[Data['ed_diagnosis'] == 'sx_breathing_difficulty', 'ed_diagnosis'] = 1
Data.loc[Data['ed_diagnosis'] == 'sx_others', 'ed_diagnosis'] = 2
Data.loc[Data['ed_diagnosis'] == 'sx_flu', 'ed_diagnosis'] = 3
Data.loc[Data['ed_diagnosis'] == 'sx_fever', 'ed_diagnosis'] = 4
Data.loc[Data['ed_diagnosis'] == 'sx_cough', 'ed_diagnosis'] = 5

Data = Data.fillna(Data.median())
X = Data[['age','sex','lab_ddimer', 'lab_crp', 'lab_lymphocyte_percentage'
                , 'lab_urea', 'lab_lymphocyte', 'lab_neutrophil_percentage']]

X = X.to_numpy()
Y = Outcomes.drop(labels='PATIENT ID', axis='columns')
Y = Y.to_numpy()
Y = Y.ravel()



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

clfDecision = DecisionTreeClassifier(criterion='entropy')
Y_pred1 = clfDecision.fit(X_train, Y_train).predict(X_test)

clfRandom = RandomForestClassifier(min_samples_leaf=8, n_estimators= 300)
Y_pred_prob = clfRandom.fit(X_train, Y_train).predict_log_proba(X_test)
Y_pred2 = clfRandom.predict(X_test)
for i in range(0, Y_pred2.shape[0]):
        if Y_pred_prob[i][1] > -1:
                Y_pred2[i] = 1
        else:
                Y_pred2[i] = 0

clfNN = MLPClassifier(activation='logistic', hidden_layer_sizes=(40, 30, 15), max_iter=150000, verbose=False, tol=0.00001, 
                                n_iter_no_change=25, learning_rate_init=0.00008, batch_size=100)

Y_pred_prob1 = clfNN.fit(X_train, Y_train).predict_proba(X_test)
Y_pred3_1 = clfNN.predict(X_test)
Y_pred_prob2 = clfNN.fit(X_train, Y_train).predict_proba(X_test)
Y_pred3_2 = clfNN.predict(X_test)

for i in range(0, len(Y_pred1)):
        if Y_pred_prob1[i][0] < 0.7:
                Y_pred3_1[i] = 1
        if Y_pred_prob2[i][0] < 0.7:
                Y_pred3_2[i] = 1
        #print(Y_pred_prob1[i][0], Y_pred_prob2[i][0], Y_pred3_1[i], Y_pred3_2[i], Y_test[i])

Y_pred3 = Y_pred3_1 | Y_pred3_2

Y_pred = Y_pred2 | Y_pred3
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