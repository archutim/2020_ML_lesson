from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
from tensorflow.keras.applications.vgg16 import VGG16

# Open the image form working directory
Outcomes = pd.read_csv("cxr_label_train.csv")
X = np.array([np.array(Image.open('./IML_CXR/' + str(fname) + '.jpg'), dtype='float32') for fname in Outcomes['PATIENT ID']])
#three_d = np.array([np.repeat((np.array(Image.open('./IML_CXR/' + str(fname) + '.jpg'), dtype='float32') / 255)[:, :, np.newaxis], 3, axis=2) for fname in Outcomes['PATIENT ID']])
three_d = np.repeat(X[:, :, :, np.newaxis], 3, axis=3)
# print(three_d.shape)
# print(X[0])
Y = Outcomes['hospital_outcome']
#Y = to_categorical(Y)

X_train, X_test, Y_train, Y_test = train_test_split(three_d, Y, test_size=0.3)

rotate_X_train = np.zeros(X_train.shape)

for i in range(len(X_train)):
        rotate_X_train[i] = rotate(X_train[i], angle=randint(-90, 90), reshape=False)

X_train = np.concatenate((X_train, rotate_X_train), axis=0)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3])
seed = randint(0, 10000)
np.random.seed(seed)  
np.random.shuffle(X_train)

Y_train = np.concatenate((Y_train, Y_train), axis=0)
np.random.seed(seed)
np.random.shuffle(Y_train)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3])

input_shape = (320, 320, 3)

base_model = VGG16(input_shape = (320, 320, 3), # Shape of our images
                    include_top = False, # Leave out the last fully connected layer
                    weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False

base_model.add(Dense(512, activation='relu'))
base_model.add(Dense(512, activation='relu'))
base_model.add(Dense(1, activation='softmax'))

model = base_model
model.summary()
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['accuracy'])


# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=16, epochs=20, validation_data=(X_train, Y_train))


Y_pred = model.predict(X_test)
print(Y_pred)
# for i in range(len(Y_pred)):
#     if Y_pred[i][1] > 0.2:
#         Y_pred[i][1] = 1
#     else:
#         Y_pred[i][1] = 0
# print(Y_pred[:][1])
# TN, FN, TP, FP = 0, 0, 0, 0
# for i in range(len(Y_test)):
#         if Y_test[i][1] == 0 and Y_pred[i][1] == 0:
#                 TN += 1
#         if Y_test[i][1] == 1 and Y_pred[i][1] == 0:
#                 FN += 1
#         if Y_test[i][1] == 0 and Y_pred[i][1] == 1:
#                 FP += 1
#         if Y_test[i][1] == 1 and Y_pred[i][1] == 1:
#                 TP += 1

# print("TN:", TN, ", FN:", FN, ", TP:", TP, ", FP:", FP)
# precision, recall = (TP/(FP+TP)), (TP/(FN+TP))
# print("precision:", precision, ", recall:", recall)
# print('F1:',  2 * ((precision*recall)/(precision+recall)))