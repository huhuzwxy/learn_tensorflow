import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#变量初始化
learning_rata = 0.001
training_iter = 20000
batch_size = 128
num_inputs = 28 #每行大小
num_steps = 28 #多少行
num_hidden = 128
num_classes = 10

#数据
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

#placeholder
x = tf.placeholder(tf.float32, [None, num_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_classes])

#weight和biases定义
weights = {
    'in' : tf.Variable(tf.random_normal([num_inputs, num_hidden])),
    'out' : tf.Variable(tf.random_normal([num_hidden, num_classes]))
}

biases = {
    'in' : tf.Variable(tf.constant(0.1, shape = [num_hidden, ])),
    'out' : tf.Variable(tf.constant(0.1, shape = [num_classes, ]))
}

#定义rnn结构
def rnn(X, weights, biases):
    #input layers
    X = tf.reshape(X, [-1, num_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, num_steps, num_hidden])
    #cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias = 1.0, state_is_tuple = True)
    init_state = lstm_cell.zero_state(batch_size, dtype = tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state = init_state, time_major = False)
    #output layers
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results

#输出
pred = rnn(x, weights, biases)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) 更新，使用方法如下
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
train = tf.train.AdamOptimizer(learning_rata).minimize(cost)

#准确率
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuarcy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#初始化变量
init = tf.global_variables_initializer()

#训练
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iter:
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        x_batch = x_batch.reshape([batch_size, num_steps, num_inputs])
        sess.run([train], feed_dict = {x : x_batch, y : y_batch})
        if step % 20 == 0:
            print(sess.run(accuarcy, feed_dict = {x : x_batch, y : y_batch}))
        step += 1

