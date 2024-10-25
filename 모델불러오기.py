import tensorflow as tf
import keras

model = keras.models.load_model('./fashion_model')

#model.load_weights('해당 가중치 파일')
model.summary()
