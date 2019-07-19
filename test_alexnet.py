# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:17:50 2019

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
#第一层卷积网络，使用96个卷积核，大小为11x11步长为4， 要求输入的图片为227x227， 3个通道，不加边，激活函数使用relu
model.add(Conv2D(96, (5, 5), strides=(1, 1), input_shape=(28, 28, 1), padding='same', activation='relu',
                 kernel_initializer='uniform'))
# 池化层
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
#使用池化层，步长为2
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 第三层卷积，大小为3x3的卷积核使用384个
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# 第四层卷积,同第三层
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# 第五层卷积使用的卷积核为256个，其他同上
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.load_weights('alexnet__model.h5')

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