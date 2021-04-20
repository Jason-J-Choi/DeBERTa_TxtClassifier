
import pathlib
import torch
import tensorflow as tf

torch.cuda.set_device(0)
device_name = "cuda:0"
device = torch.device(device_name)

torch.backends.cudnn.benchmark = False

tf.compat.v1.disable_eager_execution()
file_path = pathlib.Path.cwd()

