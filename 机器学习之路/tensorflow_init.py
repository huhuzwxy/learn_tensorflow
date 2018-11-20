#莫烦python tensorflow基础架构
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #cpu警告

#例子2
x = np.random.rand(100).astype(np.float32)
y = x * 0.3 + 0.1

weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #定义数值/数组等，设置范围
biases = tf.Variable(tf.zeros([1]))
y_pred = x * weights + biases
loss = tf.reduce_mean(tf.square(y - y_pred))
optimizer = tf.train.GradientDescentOptimizer(0.5) #梯度下降
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer() #变量初始化
sess.run(init)
print(sess.run(weights))
print(sess.run(biases))
for step in range(201):
    sess.run(train)
    if step%20 == 0:
        print('step', step,'loss', sess.run(loss),'wights', sess.run(weights),'biases', sess.run(biases))

#session会话控制
mat1 = tf.constant([[2,2]])
mat2 = tf.constant([[3], [3]])
mul = tf.matmul(mat1,mat2)

with tf.Session() as sess:
    print('mat1', sess.run(mat1))  #output语句并未执行，只是结构，sess.run()后执行
    print('mat2', sess.run(mat2))
    print('output', sess.run(mul))

#Variable变量
state = tf.Variable(0, name = 'var') #定义变量的值和名字
con = tf.constant(1)
add = tf.add(state, con)
update = tf.assign(state, add)

init = tf.global_variables_initializer() #变量一定要初始化

with tf.Session() as sess:
    sess.run(init)
    for step in range(5):
        print('step', step, 'state', sess.run(state))
        print('step', step, 'con', sess.run(con))
        print('step', step, 'add',sess.run(add))
        print('step', step, 'update',sess.run(update))

#Placeholder传入值
input1 = tf.placeholder(tf.float32, shape = (1,2))
input2 = tf.placeholder(tf.float32, shape = (2,1))
output = tf.matmul(input1, input2)  #两个数值相乘时用multiply

with tf.Session() as sess:
    in1 = np.array([[7,2]])  #噗 创建矩阵的时候两个[]，一行的矩阵时不要忘记
    in2 = np.array([[7],[2]])
    print(in1)
    print(in2)
    print(sess.run(output, feed_dict = {input1: in1, input2: in2}))

#激励函数 线性函数无法解决时考虑用非线性函数

