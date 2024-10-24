import tensorflow as tf
from keras.optimizers import Adam
import random
import keras

# 훈련용 데이터
x_train = [160, 165, 170, 175, 180]
y_train = [40, 47, 50, 70, 80]

# 모델 변수(w값) 초기화
a = tf.Variable(random.random)
b = tf.Variable(random.random)

# 옵티마이저 설정
# 학습률, 최적화 알고리즘 설정 (w값 몇씩 증가할거냐)
opt = Adam(learning_rate=0.5)

#손실함수
def loss_func():
  #예측 모델 생성
  y_pred = x_train * a + b
  return keras.losses.mse(y_train, y_pred)
 
#모델학습
epochs = 10000 #학습횟수
#opt.minimize(손실함수, var_list=변경할값)
for i in range(epochs):
  opt.minimize(loss_func, var_list=[a,b])
  if i % 1000 == 0:
    print(f'epoch:{i} : a={a.numpy()} b={b.numpy()} loss={loss_func().numpy()}')
    #print(a.numpy(),b.numpy())

y_pred = x_train * a +b
print(f'예측 결과값 : {y_pred.numpy()}')
print(f'원본 결과값: {y_train}')
  