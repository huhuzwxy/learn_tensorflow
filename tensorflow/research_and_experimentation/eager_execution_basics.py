import tensorflow as tf
from tensorflow.python.client import device_lib
import tempfile

#用最新版本的tensorflow,用Eager Execution可以完成交互式
tf.enable_eager_execution()
tf.executing_eagerly()

a = tf.add(1, 2)
print(a)

#tensor张量

#gpu加速
#是否有gpu可用
tf.test.is_gpu_available()
#列出可用设备
print(device_lib.list_local_devices())
#加速
with tf.device('CPU:0'):
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith('CPU:0')

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
_, filename = tempfile.mkstemp()
print(filename)
with open(filename, 'w') as f:
    f.write("""Line 1
    Line2
    Line3
    """)
ds_file = tf.data.TextLineDataset(filename)
print(ds_file)
