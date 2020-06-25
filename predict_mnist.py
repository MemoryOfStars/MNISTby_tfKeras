# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 19:51:51 2020

@author: 75100
"""

from tensorflow import keras

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

'''
读取bmp文件并转换成npz文件
'''
path = "./prediction.bmp"
list_img_3d = []
img = Image.open(path).convert("L")
list_img_3d.append(np.array(img))
arr_img_3d = np.array(list_img_3d)
np.savez("test_b9.npz", vol = arr_img_3d)


'''
1.load keras model
2.load npz file
3.reverse color
4.show npz image
5.recognize this brand new npz image
'''
model = keras.models.load_model('mnist-model.h5')
print("before load")
prediction_data = np.load('test_b9.npz')['vol']
prediction_data = 255-prediction_data
print("after load")
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(prediction_data[0], cmap=plt.cm.gray)
print(model.predict(np.expand_dims(prediction_data,axis=3)))