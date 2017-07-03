import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import subprocess


def total_gpu_memory():
    output = subprocess.check_output(["nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"], shell=True)
    memories = list(map(int, output.splitlines()))
    total_memory = 0

    for memory in memories:
        total_memory += memory

    return float(total_memory)


# required_memory: MiB
def use_gpu(required_memory=230):
    fraction = required_memory / total_gpu_memory()
    print("Allocating per_process_gpu_memory_fraction: {0}".format(fraction))
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    set_session(tf.Session(config=config))