#莫烦python tensorflow分类及过拟合
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

#数据
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
#mnist = read_data_sets('MNIST_data', one_hot = True)
#print(mnist)

#导入数据
x_input = tf.placeholder(tf.float32)
y_input = tf.placeholder(tf.float32)

#层设置
def add_layer(input, insize, outsize, activation_function):
    weights = tf.Variable(tf.random_normal([insize, outsize]))
    biases = tf.Variable(tf.zeros([1, outsize]) + 0.1)
    wx_plus_b = tf.matmul(input, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs

#准确率
def accuracy(x_pred, y_real):
    global pred
    prediction = sess.run(pred, feed_dict = {x_input: x_pred})
    correct_accuracy = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_real, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_accuracy, 'float'))
    result = sess.run(accuracy, feed_dict = {x_input: x_pred, y_input: y_real})
    return result

#输出层
pred = add_layer(x_input, 784, 10, activation_function = tf.nn.softmax)
loss = tf.reduce_mean(-tf.reduce_sum(y_input * tf.log(pred), reduction_indices = [1])) #loss用交叉熵函数来衡量相似度
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#session会话及变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        x_batch, y_batch = mnist.train.next_batch(100)
        sess.run(train, feed_dict = {x_input: x_batch, y_input: y_batch})
        if i % 50 == 0:
            #print(sess.run(loss, feed_dict = {x_input: x_batch, y_input: y_batch}))
            print(accuracy(mnist.test.images, mnist.test.labels))

