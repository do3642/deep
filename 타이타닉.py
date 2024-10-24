#pip install seaborn
import seaborn as sns
import tensorflow as tf
import keras
import pandas as pd

titanic = sns.load_dataset('titanic')
print(titanic.head())




titanic['age'].fillna(titanic['age'].mean(),inplace=True)
titanic['embarked'].fillna(titanic['deck'].mode()[0],inplace=True)
titanic['embark_town'].fillna(titanic['deck'].mode()[0],inplace=True)
titanic['deck'].fillna(titanic['deck'].mode()[0],inplace=True)
#print(titanic.isnull().sum())

x_train = titanic[['pclass','sex','age','who']].values
y_train = titanic['survived'].values

#print(x_train)
#print(y_train)
#print(type(x_train))
#print(type(y_train))

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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train, epochs=100)


