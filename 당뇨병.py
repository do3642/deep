import pandas as pd
import keras
import numpy as np

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# 데이터셋에 열 이름이 없어서 세팅해줘야 함
# [임신 횟수, 포도당 농도, 혈압, 피부 두께, 인슐린 수치, BMI, 당뇨 가계력, 나이, 당뇨 여부]
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# 데이터셋 로드
data = pd.read_csv(url, names=columns)

# x_train = data[[
#   'Pregnancies',
#   'Glucose',
#   'BloodPressure',
#   'SkinThickness',
#   'Insulin', 
#   'BMI',
#   'DiabetesPedigreeFunction',
#   'Age'
# ]].values


x_train = data.drop(columns=['Outcome']).values
y_train = data['Outcome'].values

model = keras.models.Sequential([
  keras.layers.Dense(1024, activation='relu'),
  keras.layers.Dense(512, activation='relu'),
  keras.layers.Dense(256, activation='relu'),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(32, activation='relu'),
  keras.layers.Dense(16, activation='relu'),
  keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

model.fit(x_train,y_train, epochs=50)

# [임신 횟수, 포도당 농도, 혈압,
#  피부 두께, 인슐린 수치, BMI, 당뇨 가계력, 나이]
# 1: 0~17  / 2: 0~191 / 3: 20~122 /  4:0~99 / 5:0~255 / 6: 0~46.3 / 7: 0.171~1.353 / 8: 나이
#predictions = model.predict([[15,120,80,40,120,25,0.5,24]])
sample = np.array([
  [6,148,72,35,0,33.6,0.627,50],
  [1,85,66,29,0,26.6,0.351,31]

])
predictions = model.predict(sample)
print(f"결과 : {predictions}")