#在交叉验证集上，准确率到达某个峰值后开始下降，则考虑模型可能出现了过拟合
#解决过拟合方法：（1）更大数据集；（2）减小网络模型；（3）正则化：权重正则化；（4）dropout

import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

#加载数据集
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

#转换为multi hot encoding
def multi_hot_encoding(train_data, num_words):
    result = np.zeros((len(train_data), num_words))
    for i, word_indices in enumerate(train_data):
        #enumerate用法：遍历列表生成索引形式
        #print('i = {}, word_indices = {}'. format(i, word_indices))
        result[i, word_indices] = 1.0
    return result
train_data = multi_hot_encoding(train_data, 10000)
print('train_data shape:', train_data.shape)
test_data = multi_hot_encoding(test_data, 10000)

#调试不同模型
def model(output_shape):
    base_model = keras.Sequential([
        keras.layers.Dense(output_shape, activation = tf.nn.relu, input_shape = (10000, )),
        keras.layers.Dense(output_shape, activation = tf.nn.relu),
        keras.layers.Dense(1, activation = tf.nn.sigmoid)
    ])
    base_model.summary()
    base_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', 'binary_crossentropy'])
    history = base_model.fit(train_data, train_labels, epochs = 20, verbose = 2, validation_data = (test_data, test_labels), batch_size = 512)
    return history
#基本模型
base_history = model(16)
#small模型
small_history = model(4)
#big模型
big_history = model(512)

#结果可视化
def view(histories):
    plt.figure()
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_binary_crossentropy'], '--', label = name.title() + 'val')
        train = plt.plot(history.epoch, history.history['binary_crossentropy'], color = val[0].get_color(), label = name.title() + 'train')
    plt.xlabel('epochs')
    plt.ylabel('bianry_crossentropy')
    plt.legend()
    plt.show()
view([('base_model', base_history),
      ('small_model', small_history),
      ('big_model', big_history)
      ])

#l2正则化模型
def model(output_shape):
    base_model = keras.Sequential([
        keras.layers.Dense(output_shape, kernel_regularizer = keras.regularizers.l2(0.001), activation = tf.nn.relu, input_shape = (10000, )),
        keras.layers.Dense(output_shape, kernel_regularizer = keras.regularizers.l2(0.001), activation = tf.nn.relu),
        keras.layers.Dense(1, activation = tf.nn.sigmoid)
    ])
    base_model.summary()
    base_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', 'binary_crossentropy'])
    history = base_model.fit(train_data, train_labels, epochs = 20, verbose = 2, validation_data = (test_data, test_labels), batch_size = 512)
    return history
l2_base_history = model(16)
view([('base_model', base_history),
      ('l2_base_model', l2_base_history)])

#dropout，通常值设置为0.2至0.5
def model(output_shape):
    base_model = keras.Sequential([
        keras.layers.Dense(output_shape, kernel_regularizer = keras.regularizers.l2(0.001), activation = tf.nn.relu, input_shape = (10000, )),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_shape, kernel_regularizer = keras.regularizers.l2(0.001), activation = tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation = tf.nn.sigmoid)
    ])
    base_model.summary()
    base_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', 'binary_crossentropy'])
    history = base_model.fit(train_data, train_labels, epochs = 20, verbose = 2, validation_data = (test_data, test_labels), batch_size = 512)
    return history
dropout_base_history = model(16)
view([('base_model', base_history),
      ('dropout_base_model', dropout_base_history)])



