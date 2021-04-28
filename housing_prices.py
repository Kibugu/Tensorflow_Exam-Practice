import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[1])])
model.compile(loss='mean_squared_error',optimizer='SGD')
model.summary

x = np.array([-1,0,1,2,3,4])
y = np.array([-3,-1,1,3,5,7])

model.fit(x,y,epochs = 600)
y_pred = model.predict([10])


print(x)
print(y)
print(y_pred)