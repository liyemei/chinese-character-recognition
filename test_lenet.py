# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:31:17 2019

@author: liyemei
"""
from __future__ import print_function
import os
import keras
from keras.callbacks import History 
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Model, load_model
from keras.callbacks import History 
data_dir = './data'
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')
from keras.models import Sequential
from keras.layers import Dense 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.load_weights('lenet__model.h5')

img = Image.open(os.path.join('images', '1166.png'))
plt.figure("Image") # 图像窗口名称
plt.imshow(img)
plt.axis('on') # 关掉坐标轴为 off
plt.title('chinese char') # 图像题目
plt.show()
testData = img.resize(( 28, 28))
testData=np.array(testData .convert('L' ))

testData=testData.reshape(1,28,28,1)
pro=model.predict(testData)
print ('识别概率：{}'.format(pro))
maxindex  = np.argmax(pro)
from xpinyin import Pinyin

char_dict= {'一': 0, '丁': 1, '七': 2, '万': 3, '丈': 4, '三': 5, '上': 6, '下': 7, '不':8 ,'与':9}
pin = Pinyin()

for key, val in char_dict.items():
    if val == maxindex :
        print("识别结果为：{}".format(key))
        print("汉字拼音为：{}".format(pin.get_pinyin(key) ))