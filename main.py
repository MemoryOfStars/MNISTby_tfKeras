# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 20:36:22 2020

@author: stardust_memory
"""


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

mnist = keras.datasets.mnist

def get_train_val(mnist_path):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data(mnist_path)
    print("train_images nums:{}".format(len(train_images)))
    print("test_images nums:{}".format(len(test_images)))
    return train_images, train_labels, test_images, test_labels

'''
display 25 images and labels
'''
def show_mnist(images, labels):
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.xlabel(str(labels[i]))
    plt.show()
    
    
def one_hot(labels):
    onehot_labels=np.zeros(shape=[len(labels),10])
    for i in range(len(labels)):
        index = labels[i]
        onehot_labels[i][index]=1
    return onehot_labels

def mnist_net(input_shape):
    '''
    A easy full-connected network
    input_layer:28*28=784
    hidden_layer:120
    output_layer:10
    '''
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.Dense(units=120, activation=tf.nn.relu))
    model.add(keras.layers.Dense(units=10, activation=tf.nn.softmax))
    return model


def mnist_cnn(input_shape):
    '''
    A CNN network
    :param input_shape:input dimension
    :return:CNN model
    '''
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32,kernel_size = 5, strides = (1,1),
                                  padding = 'same', activation = tf.nn.relu, input_shape = input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding = 'valid'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides = (1,1),
                                  padding = 'same',activation = tf.nn.relu))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2), padding = 'valid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=10, activation = tf.nn.softmax))
    return model


def train_model(train_images, train_labels, test_images, test_labels):
    train_images=train_images/255.0
    test_images = test_images/255.0
    
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)
    print("train_images :{}".format(train_images.shape))
    print("test_images :{}".format(test_images.shape))
    train_labels=one_hot(train_labels)
    test_labels=one_hot(test_labels)
    
    
    logdir = os.path.join("log")    # 记录回调tensorboard日志打印记录
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    model = mnist_cnn(input_shape=(28,28,1))
    model.compile(optimizer=tf.optimizers.Adam(),loss="categorical_crossentropy",metrics=['accuracy'])
    model.fit(x=train_images, y=train_labels, epochs=5,callbacks=[tensorboard_callback])
    
    test_loss,test_acc=model.evaluate(x=test_images, y=test_labels)
    print("Test Accuracy %.2f"%test_acc)
    
    
    cnt=0
    predictions=model.predict(test_images)
    for i in range(len(test_images)):
        target=np.argmax(predictions[i])
        label=np.argmax(test_labels[i])
        if target==label:
            cnt += 1
            
    print("correct prediction of total: %.2f"%(cnt/len(test_images)))
    
    model.save('mnist-model.h5')
    
if __name__ == '__main__':
    mnist_path = 'mnist.npz'
    train_images, train_labels, test_images, test_labels=get_train_val(mnist_path)
    #show_mnist(train_images[500:],train_labels[500:])
    
    train_model(train_images, train_labels, test_images, test_labels)
    


























