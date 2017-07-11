import os

def use_cpu():
    """Sets keras/tensorflow to use CPU"""

    os.environ["CUDA_VISIBLE_DEVICES"] = ''

