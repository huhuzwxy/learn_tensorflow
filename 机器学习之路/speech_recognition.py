import tensorflow as tf

#变量初始化
learning_rate = 0.001
momentum = 0.9
num_epoches = 40
batch_size = 10
num_hidden = 100
num_layer = 1
num_features = 13
num_classes = ord('z') - ord('a') + 1 + 1 + 1

graph = tf.Graph()
with graph.as_default():
    inputs = tf.placeholder(tf.float32, [None, None, num_features])
    targets = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [None])

    cell = tf.contrib.rnn.LSTMCell()


input = tf.placeholder(tf.float32, [None, None, num_features])
targets = tf.sparse_placeholder(tf.int32)
seq_len = tf.placeholder(tf.int32, [None])

with tf.Session() as sess:

