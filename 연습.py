import tensorflow as tf
from keras.optimizers import Adam
import random
import keras

x_train = [1,2,3,4,5]
y_train = [15,25,35,45,55]

w1 = tf.Variable(random.random)
w2 = tf.Variable(random.random())

opt = Adam(learning_rate=0.1)

def loss_func():
  y_pred = x_train * w1 + w2
  return keras.losses.mse(y_train,y_pred)

epochs = 1000
for i in range(epochs):
  opt.minimize(loss_func, var_list=[w1,w2])
  print(w1.numpy(),w2.numpy())

y_pred = x_train * w1 +w2
print(y_pred.numpy())