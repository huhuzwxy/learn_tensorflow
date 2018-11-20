import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import os

#数据
mnist = keras.datasets.mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
print('train_data shape:{}, train_labels shape:{}'. format(train_data.shape, train_labels.shape))
print('test_data shape:{}, test_labels shape:{}'. format(test_data.shape, test_labels.shape))
plt.figure()
plt.imshow(train_data[1])
plt.show()
train_data = train_data[: 1000].reshape(-1, 28 * 28) / 255.0
test_data = test_data[: 1000].reshape(-1, 28 * 28) / 255.0
train_labels = train_labels[: 1000]
test_labels = test_labels[: 1000]
print('train_data shape:{}, train_labels shape:{}'. format(train_data.shape, train_labels.shape))

#模型
def creat_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.sparse_categorical_crossentropy, metrics = ['accuracy'])
    model.summary()
    return model

#checkpoint使用
#checkpoint_path = 'training_1/cp.ckpt'
checkpoint_path = 'training_2/cp-{epoch: 04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
#cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True, verbose = 1)
#checkpoint使用，period = 5 为每5个epoch保存一次
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True, verbose = 1, period = 5)
model = creat_model()
model.fit(train_data, train_labels, epochs = 30, validation_data = (test_data, test_labels), callbacks = [cp_callback])

#创建新模型使用checkpoint
model = creat_model()
#未训练模型效果
loss, acc = model.evaluate(test_data, test_labels)
print('未训练的模型loss:{},未训练的模型acc:{}'. format(loss, acc))
#用checkpoint保存的权重更新新模型
#model.load_weights(checkpoint_path)
#loss, acc = model.evaluate(test_data, test_labels)
#print('用checkpoint更新模型loss:{},用checkpoint更新模型acc:{}'. format(loss, acc))
#用period = 5的最后一个epoch更新权重
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
loss, acc = model.evaluate(test_data, test_labels)
print('用period = 5的最后一个epoch更新模型loss:{},用period = 5的最后一个epoch更新模型acc:{}'. format(loss, acc))

#手动保存权重
model.save_weights('./checkpoints/my_checkpoint')
model = creat_model()
model.load_weights('./checkpoints/my_checkpoint')
loss, acc = model.evaluate(test_data, test_labels)
print('用手动保存的权重更新模型loss:{},用p手动保存的权重更新模型acc:{}'. format(loss, acc))

#保存和重新加载整个模型
model = creat_model()
model.fit(train_data, train_labels, epochs = 5)
model.save('my_model.h5')
#重新加载
new_model = keras.models.load_model('my_model.h5')
new_model.summary()



