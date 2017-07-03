from keras.backend.tensorflow_backend import set_session
import tensorflow as tf


def use_cpu():
    set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))