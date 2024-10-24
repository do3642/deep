import tensorflow as tf
import keras
import pandas as pd

data = pd.read_csv('./Admission_Data.csv')
#결측값 제거
data.dropna(inplace=True)
#print(data.isnull().sum())

x_train = data[['GPA','Age','English_Score']].values
y_train = data['Admission'].values
#print(x_train)
#print(type(y_train))

#exit() #아래를 먼저써서 위쪽 실행후 마감하는역할

#sequential은 순차적으로 쌓음
#숫자는 노드개수 옆에 옵션은 활성함수
#모델생성
model = keras.models.Sequential([
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(32, activation='relu'),
  keras.layers.Dense(1, activation='sigmoid'),
])

#밑에 옵티마이저를 직접 생성 및 설정해서 사용하는 법
#opt = keras.optimizers.Adam(learning_rate=0.001)

#모델 학습전 설정
# metrice는 평가지표, accuracy는 정확도
#binary_crossetropy는 이진분류할때 쓰는 손실함수
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 학습
model.fit(x_train,y_train, epochs=500)

# 학습이 완료된 모델을 이용해서 합격여부 결과 예측
predictions = model.predict([[4.5, 20, 100],[1.2, 25, 70]])
print(predictions)