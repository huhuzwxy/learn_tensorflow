#利用keras对衣服进行分类
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt
import numpy as np

#加载数据集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#可视化
print('train images shape：', train_images.shape)
print('length of train labels:', len(train_labels))
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#plt.figure(figsize = [28, 28])
#for i in range(10):
#    plt.subplot(2, 5, i + 1)
#    plt.imshow(train_images[i])
#    plt.show()
#判断数据是否平衡
labels_sum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(train_labels)):
    for j in range(10):
        if (train_labels[i] == j):
            labels_sum[j] = labels_sum[j] + 1
print('labels_sum:', labels_sum)

#数据归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

#分类模型
model = keras.Sequential([
    #输入图片转化为一维数组
    keras.layers.Flatten(input_shape = (28, 28)),
    #神经网络层
    keras.layers.Dense(128, activation = tf.nn.relu),
    #softmax层，分为10类
    keras.layers.Dense(10, activation = tf.nn.softmax)
])
#keras.Sequential配置训练模型
model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#keras.Sequential训练模型
model.fit(train_images, train_labels, epochs = 5)
#keras.Sequential验证训练模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_loss:', test_loss, 'test_acc:', test_acc)
#keras.Sequential预测
predict = model.predict(test_images)
print('length of predict:', len(predict))
print('predict[2]:', predict[2], 'max of predict[2]:', np.argmax(predict[2]), 'test_labels of predict[2]:', test_labels[2])


