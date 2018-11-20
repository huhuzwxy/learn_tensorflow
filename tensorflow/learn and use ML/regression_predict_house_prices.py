import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#加载数据
boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

#数据可视化
print('train_data shape:{}, test_data shape:{}'.format(train_data.shape,test_data.shape))
print(train_data.shape[1])
#利用pandas可视化
def view(train_data, train_labels):
    column_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', ' dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
    table0 = pd.DataFrame(train_data, columns = column_names)
    column_name = ['labels']
    table1 = pd.DataFrame(train_labels, columns = column_name)
    table = pd.concat([table0, table1], axis = 1)
    #axis = 0纵向合并，axis = 1横向合并
    print(table.head())
view(train_data, train_labels)

#数据预处理
mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
view(train_data, train_labels)

#keras分类模型
model = keras.Sequential([
    keras.layers.Dense(64, activation = tf.nn.relu, input_shape = (train_data.shape[1],)),
    keras.layers.Dense(64, activation = tf.nn.relu),
    keras.layers.Dense(1)
])
model.summary()
model.compile(optimizer = tf.train.RMSPropOptimizer(0.001), loss = 'mse', metrics = ['mae'])
history = model.fit(train_data, train_labels, epochs = 100, validation_split = 0.3, verbose = 0)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('test_loss:{}, test_acc:{}'. format(test_loss, test_acc))
prediction = model.predict(test_data)

#结果可视化
def view_result(history):
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('mean abs error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label = 'train loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label = 'val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()
view_result(history)

def view_predict(test_labels, prediction):
    plt.figure()
    plt.xlabel('true values')
    plt.ylabel('prediction')
    plt.scatter(test_labels, prediction)
    plt.show()
view_predict(test_labels, prediction)




