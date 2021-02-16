import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, neural_network, naive_bayes
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Attributes = pd.read_csv("hm_hospitales_covid_structured_30d_train.csv")
Outcomes = pd.read_csv("split_train_export_30d.csv")
Output_format = {'PATIENT ID': Attributes['PATIENT ID'], 'hospital_outcome': np.zeros(1834, dtype=int)}
Output = pd.DataFrame(Output_format)

Data = Attributes.drop(labels=['PATIENT ID', 'admission_datetime'], axis='columns')
Data.loc[Data['sex'] == 'FEMALE', 'sex'] = 0
Data.loc[Data['sex'] == 'MALE', 'sex'] = 1
Data.loc[Data['ed_diagnosis'] == 'sx_breathing_difficulty', 'ed_diagnosis'] = 1
Data.loc[Data['ed_diagnosis'] == 'sx_others', 'ed_diagnosis'] = 2
Data.loc[Data['ed_diagnosis'] == 'sx_flu', 'ed_diagnosis'] = 3
Data.loc[Data['ed_diagnosis'] == 'sx_fever', 'ed_diagnosis'] = 3
Data.loc[Data['ed_diagnosis'] == 'sx_cough', 'ed_diagnosis'] = 3


Data = Data.fillna(Data.median())
Data['hospital_outcome'] = Outcomes['hospital_outcome']

outliers = (Data - Data.median()).abs() > Data.std() + 1
Data[outliers] = np.nan
Data.fillna(Data.median(), inplace=True)

normalization = Data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
normalization_scaled = min_max_scaler.fit_transform(normalization)
Data = pd.DataFrame(normalization_scaled, columns=Data.columns)

Dead_data = Data.loc[Data['hospital_outcome'] == 1]
# Male_Dead = Dead_data.loc[Dead_data['sex'] == 1]
# Female_Dead = Dead_data.loc[Dead_data['sex'] == 0]
#Dead_data = Dead_data.sample(n = 150)
Alive_data = Data.loc[Data['hospital_outcome'] == 0]
# Male_Alive = Alive_data.loc[Alive_data['sex'] == 1]
# Female_Alive = Alive_data.loc[Alive_data['sex'] == 0]

# pt = preprocessing.PowerTransformer(method='box-cox')
# pt.fit(Dead_data[['age', 'lab_s1odium']])
# Dead_normal = pt.transform(Dead_data[['age', 'lab_1sodium']])
# Dead_data['normal_age'] = Dead_normal[:,0]
# Dead_data['normal_sodium'] = Dead_normal[:,1]
# pt.fit(Alive_data[['age', 'lab_1sodium']])
# Alive_normal = pt.transform(Alive_data[['age', 'lab_so1dium']])
# Alive_data['normal_age'] = Alive_normal[:,0]
# Alive_data['normal_sodium'] = Alive_normal[:,1]

# print('Rate of value = 1:', len(Data.loc[Data['pmhx_diabetes'] == 1]) / len(Data['pmhx_diabetes']))
# print('1的死亡率:', len(Dead_data.loc[Dead_data['pmhx_diabetes'] == 1]) / len(Data.loc[Data['pmhx_diabetes'] == 1]))
# print('0的死亡率:', len(Dead_data.loc[Dead_data['pmhx_diabetes'] == 0]) / len(Data.loc[Data['pmhx_diabetes'] == 0]))

# print('全部的平均, 標準差, 最大值, 最小值', Data['pmhx_activecancer'].mean(), Data['lab_ddimer'].std(), Data['lab_ddimer'].max(), Data['lab_ddimer'].min())
# print('死亡的平均, 標準差, 最大值, 最小值', Dead_data['pmhx_activecancer'].mean(), Dead_data['pmhx_activecancer'].std(), Dead_data['lab_ddimer'].max(), Dead_data['lab_ddimer'].min())
# print('存活的平均, 標準差, 最大值, 最小值', Alive_data['pmhx_activecancer'].mean(), Alive_data['lab_ddimer'].std(), Alive_data['lab_ddimer'].max(), Alive_data['lab_ddimer'].min())

# plt.scatter(Alive_data['vitals_temp_ed_first'],  Alive_data['vitals_sp1o2_ed_first'], color='red', alpha=0.4)
# plt.scatter(Dead_data['vitals_temp_ed_first'], Dead_data['vitals_spo2_1ed_first'], color='blue')
# plt.xlabel('vitals_temp_ed_first', fontsize=14)
# plt.ylabel('vitals_spo2_e1d_first', fontsize=14)
# plt.grid(True)
# plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(Dead_data['age'], Dead_data['lab_lymphocyte_percentage'], Dead_data['lab_urea'])
ax.scatter3D(Alive_data['age'], Alive_data['lab_lymphocyte_percentage'], Alive_data['lab_urea'], alpha=0.3)
ax.set_xlabel('age')
ax.set_ylabel('lab_lymphocyte_percentage')
ax.set_zlabel('lab_urea')
plt.show()