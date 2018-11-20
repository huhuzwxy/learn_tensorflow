import tensorflow as tf
from tensorflow import keras

#加载数据集
imdb = keras.datasets.imdb
#你大爷的，记得选num_words与后面对应
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
#数据可视化
print('length of train_data:', len(train_data))
print('length of test_data:', len(test_data))
labels_sum = [0, 0]
for i in range(len(train_data)):
    if (train_labels[i] == 0):
        labels_sum[0] = labels_sum[0] + 1
    elif (train_labels[i] == 1):
        labels_sum[1] = labels_sum[1] + 1
print('labels_sum:', labels_sum)
#不同评论长度不同
print('length of train_data[1]:{}, length of train_data[2]:{}'. format(len(train_data[1]), len(train_data[2])))

#数据预处理
#max_length_train = 0
#max_length_test = 0
#max_length = 0
#for i in range(len(train_data)):
#    length_train = len(train_data[i])
#    length_test = len(test_data[i])
#    if (length_train > max_length_train):
#        max_length_train = length_train
#    if (length_test > max_length_test):
#        max_length_test = length_test
#if (max_length_train > max_length_test):
#    max_length = max_length_train
#else:
#    max_length = max_length_test
#print('max_length:', max_length)

word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
#print(decode_review(train_data[1]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index['<PAD>'], padding= 'post', maxlen = 256)
#序列长度小于maxlen的，被填充；大于maxlen的，被截断
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index['<PAD>'], padding= 'post', maxlen = 256)
print(len(train_data))

part_train_data = train_data[10000: ]
val_train_data = train_data[ : 10000]
print(part_train_data.shape, val_train_data.shape)
part_train_labels = train_labels[10000: ]
val_train_labels = train_labels[ : 10000]
print(part_train_labels.shape, val_train_labels.shape)

#keras分类模型
model = keras.Sequential([
        keras.layers.Embedding(10000, 16),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16, activation = tf.nn.relu),
        keras.layers.Dense(1, activation = tf.nn.sigmoid)
])
model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(part_train_data, part_train_labels, epochs = 40,  batch_size = 512, validation_data = (val_train_data, val_train_labels), verbose = 1)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('test_loss:{}, test_acc:{}'. format(test_loss, test_acc))







