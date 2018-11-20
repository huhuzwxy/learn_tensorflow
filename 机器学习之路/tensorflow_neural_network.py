#莫烦python tensorflow第一个神经网络
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#构造一个神经层函数（w,b,激励函数)
def add_layer(inputs, in_size, out_size, activation_function):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases

    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)

    return outputs

#数据
x = np.linspace(-1 ,1, 300, dtype = np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x.shape)
y = np.square(x) - 0.5 + noise

#节点 读入数据
x_input = tf.placeholder(tf.float32)
y_input = tf.placeholder(tf.float32)

#隐藏层和输出层
mid1 = add_layer(x_input, 1, 10, activation_function = tf.nn.relu)
pred = add_layer(mid1, 10, 1, activation_function = None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_input - pred), reduction_indices = [1]))
#reduction_indices降维处理 =[1]时按行求和， = [0]时按列求和，默认值为None，降至0维，为一个数。
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#变量初始化
init =tf.global_variables_initializer()

#绘图
plt.figure()
plt.scatter(x ,y)
#plt.show()

#session部分
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict = {x_input: x, y_input: y})
        if i % 50 == 0:
            print('step', i, 'loss', sess.run(loss, feed_dict = {x_input: x, y_input: y}))
    plt.plot(x, sess.run(pred, feed_dict = {x_input: x}))
    plt.show()
