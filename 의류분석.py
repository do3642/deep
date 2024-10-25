import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

#이미지 가져오기 ( 크기조절, 흑백으로)
img = image.load_img('./티셔츠.jpg', target_size=(28,28),color_mode='grayscale')
img_array = image.img_to_array(img)
#print(img_array.shape)

#이미지를 배열로 변환
img_array = img_array / 255.0
#print(img_array)

# 이미지를 (1,28,28,1) 형태로 변환 
img_array = np.expand_dims(img_array, axis=0)
#print(img_array.shape)

(x_train,y_train),(x_test, y_test)=keras.datasets.fashion_mnist.load_data()
# print(x_train.shape)

#matplotlib로 이미지 보기
#plt.imshow(x_train[0])
#plt.show()

#경사하강법
x_train = x_train / 255.0
x_test = x_test /255.0


#4차원으로 바꿈
#x_train = x_train.reshape(60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

#print(x_train.shape)
#print(x_test.shape)
#print(x_train[0])
#exit()


#기존에서 마지막 숫자와 activation(활성함수)이 softmax로 바뀜
#softmax는 확률분포 구할때 쓰는 활성함수
#결과값이 10개가 나오는데 [0.1,0.3,0,0.4...0] 퍼센트가 가장 높은게 그쪽 의류다 표시
model = keras.models.Sequential([
  #이미지의 특징을 추출하는 커볼루션 레이어
  #32가지의 특징, 3x3크기만큼 잘라서 , 원본사이즈를 계속 가지고 있어라
  keras.layers.Conv2D(32, (3,3), padding="same", activation='relu',input_shape=(28,28,1)), # 인풋쉐입이 28x28 였는데 1이 붙는이유는 컨볼루션레이어는 4차원으로 받고 흑백이미지는 안에 0~255가 들어가있고 컬러는 rgb가 들어가있어서 그럼 컬러일때는 3이 들어가야함
  #4차원의 input_shape(이미지개수,이미지높이,이미지너비,1(1은흑백,3은rgb))

  #풀링 레이어
  keras.layers.MaxPooling2D((2,2)),

  #마지막 dense에서 [10개]를 보고싶은데 [28]x[10]인 상태이므로
  # Flatten을 넣어 1차원으로 바꿔준다.
  keras.layers.Flatten(), #다차원을 1차원으로 변경하는 레이어 
  keras.layers.Dense(64, activation='relu'), #어떤형태로 받을지 명시 input_shape인데 처음 시작하는 레이어로 옮겨짐
  keras.layers.Dense(32, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

#이진분류가 아니므로 loss를 sparse_categorical_crossentropy로 씀
#다중분류할땐 이거나 categorical_crossentropy를 씀
#그냥 카테고리컬은 one-hot 인코딩된 라벨일때 사용
#y => [1,0,0,0,0,0,0,0,0,0] <- ont-hot 인코딩, 1이 있는자리가 0~9숫자를 가르킴,[0,0,1,0]  = 2라는 뜻 (categorical_crossentropy) 
#y => 0~9 : 고유 정수값 (sparse_categorical_crossentropy)
model.compile(optimizer="adam" ,loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#모델 정보 출력
#위에 모델에서 input_shape이 없으면 오류나는데 이유는 처음에 어떤 형태로 받을건지 명시해줘야함
#지금은 x_train의 형식 (60000,28,28)인데 60000은 개수이므로 28,28을 넣어주면된다.
model.summary()

#model.fit(x_train,y_train, epochs=10)
# validation_data: 검증용 데이터
# 모델학습중 검증용 데이터로 검증을 함
# 모델의 과적합 현상을 모니터링
model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=10)


#모델 저장
model.save('./fashion_model')

#모델 가중치만 저장
model.save_weights('./가중치만저장')
model.load_weights('해당 가중치 파일')

# model.to_json() 모델을 json으로 


#모델 평가
#result = model.evaluate(x_test,y_test)
#print(result)

#소수 넷째자리까지만 나오게
np.set_printoptions(precision=4, suppress=True)

#이미지 예측
pred = model.predict(img_array)
print(pred)
result = np.argmax(pred)
print(f'결과: {result}')