# MNISTby_tfKeras  
----------------------------------------------------------------------------------

## main.py  
1.load MNIST dataset by keras.datasets.mnist  
2.build MNIST full-connected network and a CNN model by tf.keras  
3.train CNN model and save as mnist-model.h5  

----------------------------------------------------------------------------------

## prediction_mnist.py  
(draw a digit bmp format image first)  
1.load trained mnist-model.h5  
2.convert this bmp to npz  
3.read npz using numpy.load  
