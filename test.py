import tensorflow as tf
from tensorflow.python.client import device_lib

# if tf.test.gpu_device_name():
#   print(tf.test.gpu_device_name())

device_lib.list_local_devices()