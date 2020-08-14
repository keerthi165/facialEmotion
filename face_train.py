from sklearn.model_selection import train_test_split
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import cv2
import pickle

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

data = pd.read_csv('data.csv')
emotions = data['emotion']
pixels = data['pixels'].tolist()
#print(pixels)
faces = []
for pixel_sequence in pixels:
	face = [int(pixel) for pixel in pixel_sequence.split(' ')]
	face = np.asarray(face).reshape(48, 48)
	face = cv2.resize(face.astype('uint8'), (48,48))
	faces.append(face.astype('float32'))
faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)
classes = ["angry","disgust","scared","happy","sad","surpried","neutral"]
emotions = pd.get_dummies(data['emotion']).as_matrix()
faces = preprocess_input(faces)
x_train,x_test,y_train,y_test = train_test_split(faces,emotions,test_size = 0.25)

input_s = (48,48,1)
input = Input(input_s)
x = Conv2D(128,(1,1),strides= (2,2),padding = 'same')(input)
x = BatchNormalization()(x)
x = layers.PReLU(alpha_initializer='zeros')(x)
x = Dropout(0.5)(x)

for i in range(4):
    x = Conv2D(128,(1,1),strides=(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    x = layers.PReLU(alpha_initializer='zeros')(x)
    x = Conv2D(128,(1,1),strides=(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    x = layers.PReLU(alpha_initializer='zeros')(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)
    x = Dropout(0.5)(x)

x = Conv2D(1024,(1,1),strides=(2,2),padding='same')(x)
x = BatchNormalization()(x)
x = layers.PReLU(alpha_initializer='zeros')(x)
x = Dropout(0.5)(x)
x = Conv2D(1024,(3,3),padding='same')(x)
x = BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.5)(x)
x = Dropout(0.5)(x)

x = Conv2D(7,(3,3),padding='same')(x)
x = GlobalAveragePooling2D()(x)
output = Activation('softmax')(x)
model = Model(input,output)
model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics = ['accuracy'])
y_pred = model.predict(x_test,10)
print(model.summary())


history = model.fit(x_train, y_train,validation_split = 0.1, epochs=10, batch_size=100)
score, accu = model.evaluate(x_test,y_test)
print('Test Score:',score)
print('Test accuracy:',accu)
#model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# model.fit(x_train,y_train,batch_size = 100,epochs = 15,verbose = 1,validation_data=(x_test,y_test))
#loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('losss')
plt.xlabel('epoch')
plt.legend('train',loc='upper left')
plt.show()
#confusion matrix


print("*********************************")
print(np.asarray(y_test).argmax(axis=1))
print(y_pred.argmax(axis=1))
confusion_matrix = metrics.confusion_matrix(np.asarray(y_test).argmax(axis = 1),y_pred.argmax(axis=1))
print(confusion_matrix)
plt.imshow(confusion_matrix, interpolation="nearest",cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.show()

# filename = 'finalized_model.sav'
# pickle.dump(model,open(filename,'wb'))