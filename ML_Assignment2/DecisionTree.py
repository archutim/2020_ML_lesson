import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
# Data = Data[['age','sex', 'pmhx_diabetes', 'pmhx_hld', 'pmhx_htn', 'pmhx_ihd'
#         , 'pmhx_copd', 'pmhx_activecancer', 'pmhx_chronicliver', 'pmhx_stroke', 'pmhx_chf', 'pmhx_dementia', 'lab_ddimer'
#         , 'lab_crp', 'lab_lymphocyte_percentage', 'lab_urea', 'lab_lymphocyte', 'lab_neutrophil_percentage']]
Data = Data.fillna(Data.median())
X = Data[['age','sex', 'pmhx_diabetes', 'pmhx_hld', 'pmhx_htn', 'pmhx_ihd'
                , 'pmhx_copd', 'pmhx_activecancer', 'pmhx_chronicliver', 'pmhx_stroke', 'pmhx_chf', 'pmhx_dementia'
                ,'lab_ddimer', 'lab_crp', 'lab_lymphocyte_percentage'
                , 'lab_urea', 'lab_lymphocyte', 'lab_neutrophil_percentage']]

###Normalization
normalization = Data[['lab_ddimer', 'lab_crp', 'lab_lymphocyte_percentage'
                        , 'lab_urea', 'lab_lymphocyte', 'lab_neutrophil_percentage']].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
normalization = min_max_scaler.fit_transform(normalization)
normalization = pd.DataFrame(normalization, columns = ['lab_ddimer', 'lab_crp', 'lab_lymphocyte_percentage'
                        , 'lab_urea', 'lab_lymphocyte', 'lab_neutrophil_percentage'])
###Standardization
standardization = Data[['lab_ddimer', 'lab_crp', 'lab_lymphocyte_percentage'
                        , 'lab_urea', 'lab_lymphocyte', 'lab_neutrophil_percentage']]                        
standardization[['lab_ddimer', 'lab_crp', 'lab_lymphocyte_percentage'
                        , 'lab_urea', 'lab_lymphocyte', 'lab_neutrophil_percentage']] = StandardScaler().fit_transform(standardization)

X[['lab_ddimer', 'lab_crp', 'lab_lymphocyte_percentage', 'lab_urea', 'lab_lymphocyte', 'lab_neutrophil_percentage']] = normalization

X = X.to_numpy()
Y = Outcomes.drop(labels='PATIENT ID', axis='columns')
Y = Y.to_numpy()
Y = Y.ravel()



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

clf = DecisionTreeClassifier(criterion='entropy')
Y_pred = clf.fit(X_train, Y_train).predict(X_test)


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