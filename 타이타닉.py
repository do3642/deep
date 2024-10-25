#pip install seaborn
import seaborn as sns
import tensorflow as tf
import keras
import pandas as pd

titanic = sns.load_dataset('titanic')
print(titanic.head())




#titanic['age'].fillna(titanic['age'].mean(),inplace=True)
#titanic['embarked'].fillna(titanic['embarked'].mode()[0],inplace=True)
#titanic['embark_town'].fillna(titanic['embark_town'].mode()[0],inplace=True)
#titanic['deck'].fillna(titanic['deck'].mode()[0],inplace=True)
#print(titanic.isnull().sum())
titanic.drop(columns='deck', inplace=True)
age_mean_by_who = titanic.groupby('who')['age'].mean()

def fill_age_mean(row, mean_values):
    if pd.isnull(row['age']):
        return mean_values[ row['who'] ]
    else:
        return row['age']

# 나이 결측치를 처리하기 위한 코드
titanic['age'] = titanic.apply(fill_age_mean, axis=1, mean_values=age_mean_by_who )

titanic.dropna(inplace=True)



#print(titanic['who'].value_counts())
#print(y_train)
#print(type(x_train))
#print(type(y_train))

def changeStr(train):
  who = str(train)
  
  if "male" == who:
    return 1
  elif "female" == who:
    return 2
  elif "man" == who:
    return 1
  elif "woman" == who:
    return 2
  elif "child" == who:
    return 3

#print(titanic['sex'].value_counts())
#titanic['sex'] =titanic['sex'].apply(changeStr)
#titanic['who'] =titanic['who'].apply(changeStr)

# male : 0, female:1 로 변경
titanic['sex'] = titanic['sex'].map( { 'male': 0, 'female': 1 } )
# man : 0, woman : 1, child : 2 로 변경
titanic['who'] = titanic['who'].map( { 'man': 0, 'woman': 1, 'child': 2 } )
# S : 0, C : 1, Q : 2 로 변경
titanic['embarked'] = titanic['embarked'].map( { 'S':0, 'C':1, 'Q':2 } )



#x_train = titanic[['pclass','sex','age','who']].values

#성별에 따른 생존률 남자 19%,여자 78%
#x_train = titanic[['sex']].values
y_train = titanic['survived'].values

#등급과 성별에 따른 생존률 
# (1등급 여자 97%,2등급 여자 91%,3등급 여자 50%)
# (1등급 남자 37%,2등급 남자  14%,3등급 남자 14%)
x_train = titanic[['pclass','sex']].values


x_train = titanic[['pclass','sex','age','embarked','who']].values




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

model.fit(x_train,y_train, epochs=28)

# 'pclass','sex','age','embarked','who'

predictions = model.predict([[1,0,7,1,2],[3,0,45,0,0]])
print(f"결과 : {predictions}")