import pickle
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense,Embedding,Flatten,Bidirectional
from keras import Input,Model
from sklearn.model_selection import train_test_split

def load_glove(file):
    f = open(file,'r')
    embed = []
    word = []
    pair = dict()
    for line in f:
        split = line.split()
        word.append(split[0])
        embedding = np.array([float(val) for val in split[1:]])
        embed.append(embedding)
        pair[split[0]] = embedding
    embed = np.array(embed)
    return pair,word

df = pd.read_csv('train_data.csv')

glove = 'glove.6B.50d.txt'
glove_vec,words = load_glove(glove)

all_text = ' '.join(t for t in df['content'])
words = all_text.split()
word2num = dict(zip(words,range(len(words))))
word2num['Other'] = len(word2num)
word2num['Pad'] = len(word2num)
num2word = dict(zip(word2num.values(),word2num.keys()))
text = [[word2num[word] if word in word2num else word2num['Other']for word in content.split()] for content in df['content']]
for i,t in enumerate(text):
    if (len(t) < 500):
        text[i] = [word2num['Pad']] * (500 - len(t)) + t #append to the list the pad word's list 500-len(t) times
        #print(len(text[i]))
    elif (len(t) > 500):
        text[i] = t[:500]
        #print(len(text[i]))
    else:
        continue
x = np.array(text,dtype=object)
y = pd.get_dummies(df['sentiment']).values
#print(y)
batch_size = 100
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=42)
model = Sequential()
model.add(Embedding(len(word2num),batch_size,input_length=500))
model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(13,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer = 'Adam',metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train,batch_size=batch_size,epochs=15,validation_data=(x_test,y_test))
model.evaluate(x_test,y_test)