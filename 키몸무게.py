import tensorflow as tf
from keras.optimizers import Adam
# 가중치를 조절해서 손실값을 최소화시키게 도와주는 알고리즘

height = 180
weight = 85
# weight= height * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

#보통 0.001정도 조정하는데 지금 빨리보기 위해 0.1
opt = Adam(learning_rate=0.1)

def loss_func():
  return tf.square(weight - (height*a+b))

#opt.minimize(손실함수, var_list=변경할값)
for i in range(200):
  opt.minimize(loss_func, var_list=[a,b])
  print(a.numpy(),b.numpy())