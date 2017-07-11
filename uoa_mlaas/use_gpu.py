import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import subprocess, re, os, sys

def run_command(cmd):
    """Run command, return output as string."""

    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def gpu_memory_map(query="free"):
    """Returns map of GPU id to memory total or allocated  on that GPU."""
    output = ""
    if query is "free":
        output = run_command("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
    elif query is "total":
        output = run_command("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
    rows = output.strip().split("\n")
    return rows

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(int(memory), gpu_id) for (gpu_id, memory) in enumerate(gpu_memory_map())]
    best_memory, best_gpu = sorted(memory_gpu_map, reverse=True)[0]
    return best_gpu

# required_memory: MiB
def use_gpu(required_memory=130):
    """Sets GPU config of tensorflow session"""
 
    config = tf.ConfigProto()
    gpu_id = pick_gpu_lowest_memory()
    print("Picking GPU "+str(gpu_id))
    config.gpu_options.visible_device_list = str(gpu_id)
    fraction = required_memory / float(gpu_memory_map(query="total")[gpu_id])
    print("Allocating GPU Memory Fraction: {0}".format(fraction))
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    set_session(tf.Session(config=config))
