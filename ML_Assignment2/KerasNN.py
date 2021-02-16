import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import tensorflow.keras as keras

def custom_activation(x):
        return 2 / keras.activations.sigmoid(x)


#keras.utils.generic_utils.get_custom_objects().update({'custom_activation': Activation(custom_activation)})

def baseline_model():
	# create model
    model = keras.models.Sequential(
        [
            keras.layers.Dense(40, activation='relu', name="layer1", input_dim=9),
            keras.layers.Dense(30, activation='relu', name="layer2"),
            keras.layers.Dense(10, activation='relu', name="layer3"),
            keras.layers.Dense(2, activation='sigmoid', name="layer5")
        ]
    )         
	# Compile model #binary_crossentropy, categorical_crossentropy sparse_categorical_crossentropy
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
    return model

Attributes = pd.read_csv("hm_hospitales_covid_structured_30d_train.csv", na_values=0, na_filter=True)
Outcomes = pd.read_csv("split_train_export_30d.csv")

Output_format = {'PATIENT ID': Attributes['PATIENT ID'], 'hospital_outcome': np.zeros(1834, dtype=int)}
Output = pd.DataFrame(Output_format)

X = Attributes.drop(labels=['PATIENT ID', 'admission_datetime'], axis='columns')
X.loc[X['sex'] == 'FEMALE', 'sex'] = 1
X.loc[X['sex'] == 'MALE', 'sex'] = 2
X.loc[X['ed_diagnosis'] == 'sx_breathing_difficulty', 'ed_diagnosis'] = 1
X.loc[X['ed_diagnosis'] == 'sx_others', 'ed_diagnosis'] = 2
X.loc[X['ed_diagnosis'] == 'sx_flu', 'ed_diagnosis'] = 3
X.loc[X['ed_diagnosis'] == 'sx_fever', 'ed_diagnosis'] = 3
X.loc[X['ed_diagnosis'] == 'sx_cough', 'ed_diagnosis'] = 3

X = X.fillna(X.median())

# outliers = (X - X.median()).abs() > X.std()*3
# X[outliers] = np.nan
# X.fillna(X.median(), inplace=True)

# normalization = X.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# normalization_scaled = min_max_scaler.fit_transform(normalization)
# X = pd.DataFrame(normalization_scaled, columns=X.columns)

X = X[['age', 'sex', 'ed_diagnosis', 'lab_ddimer'
        , 'lab_crp', 'lab_lymphocyte_percentage', 'lab_urea', 'lab_lymphocyte', 'lab_neutrophil_percentage']]

X = X.to_numpy()
Y = Outcomes.drop(labels='PATIENT ID', axis='columns')
Y = Y.to_numpy()
Y = Y.ravel()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
origin_Ytest = Y_test
Y_train = keras.utils.to_categorical(Y_train)
Y_test = keras.utils.to_categorical(Y_test)
# model = keras.models.Sequential(
#     [
#         keras.layers.Dense(100, activation="sigmoid", name="layer1", input_dim=X_train.shape[1]),
#         keras.layers.Dense(60, activation="sigmoid", name="layer2"),

#         keras.layers.Dense(2, activation="sigmoid", name="layer")
#     ]
# )

# model = keras.models.Sequential()
# model.add(keras.layers.Dense(500, activation='relu', input_dim=X_train.shape[1]))
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dense(50, activation='relu'))
# model.add(keras.layers.Dense(1, activation='sigmoid'))



model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=1)

#model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=200, batch_size=50, validation_data=(X_train, Y_train))
Y_pred_prob = model.predict_proba(X_test)
print(Y_pred_prob)
Y_pred = model.predict(X_test)
# count = 0
# pred = np.ndarray(len(Y_pred), dtype=int)
# count = 0
# for i in range(len(Y_pred)):
#     if Y_pred[i][0] > Y_pred[i][1]:
#         pred[i] = 0
#     else:
#         count += 1
#         pred[i] = 1
# Y_pred = pred
Y_test = origin_Ytest

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