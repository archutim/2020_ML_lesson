from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from random import randint
from scipy.ndimage.interpolation import rotate

# Open the image form working directory
Outcomes = pd.read_csv("cxr_label_train.csv")
X = np.array([np.array(Image.open('./IML_CXR/' + str(fname) + '.jpg'), dtype='float32') for fname in Outcomes['PATIENT ID']])
Y = Outcomes['hospital_outcome']
Y = to_categorical(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

rotate_X_train = np.zeros(X_train.shape)

for i in range(len(X_train)):
        rotate_X_train[i] = rotate(X_train[i], angle=randint(0, 90), reshape=False)

X_train = np.concatenate((X_train, rotate_X_train), axis=0)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
seed = randint(0, 10000)
np.random.seed(seed)  
np.random.shuffle(X_train)

Y_train = np.concatenate((Y_train, Y_train), axis=0)
np.random.seed(seed)
np.random.shuffle(Y_train)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

model = Sequential()
# convolutional layer
model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(320,320, 1)))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(128, activation='relu'))
# output layer
model.add(Dense(2, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=16, epochs=20, validation_data=(X_train, Y_train))

Y_pred = model.predict(X_test)
print(Y_pred)
for i in range(len(Y_pred)):
    if Y_pred[i][1] > 0.3:
        Y_pred[i][1] = 1
    else:
        Y_pred[i][1] = 0
print(Y_pred[:][1])
TN, FN, TP, FP = 0, 0, 0, 0
for i in range(len(Y_test)):
        if Y_test[i][1] == 0 and Y_pred[i][1] == 0:
                TN += 1
        if Y_test[i][1] == 1 and Y_pred[i][1] == 0:
                FN += 1
        if Y_test[i][1] == 0 and Y_pred[i][1] == 1:
                FP += 1
        if Y_test[i][1] == 1 and Y_pred[i][1] == 1:
                TP += 1

print("TN:", TN, ", FN:", FN, ", TP:", TP, ", FP:", FP)
precision, recall = (TP/(FP+TP)), (TP/(FN+TP))
print("precision:", precision, ", recall:", recall)
print('F1:',  2 * ((precision*recall)/(precision+recall)))


Data_list=[x.split('.')[0] for x in os.listdir('IML_CXR_TEST')]
for i in range(len(Data_list)): 
    Data_list[i] = int(Data_list[i]) 
Data_list.sort()
Test_data = np.array([np.array(Image.open('./IML_CXR_TEST/' + str(fname) + '.jpg'), dtype='float32') for fname in Data_list])
Test_data = Test_data.reshape(Test_data.shape[0], Test_data.shape[1], Test_data.shape[2], 1)


Y_pred = model.predict(Test_data)
for i in range(len(Y_pred)):
    if Y_pred[i][1] > 0.3:
        Y_pred[i][1] = 1
    else:
        Y_pred[i][1] = 0
Outcomes = pd.DataFrame(Data_list, columns = ['PATIENT ID']) 
Outcomes['hospital_outcome'] = Y_pred[:, 1].astype(int)

Output_data = np.vstack((Data_list, Y_pred[:, 1]))
Outcomes.to_csv('107062338.csv', index=False)