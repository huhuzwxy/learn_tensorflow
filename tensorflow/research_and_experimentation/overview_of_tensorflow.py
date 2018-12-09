import tensorflow as tf

#with tf.Session() as sess:
#    a = sess.run(tf.add(3, 5))
#    print(a)

g1 = tf.get_default_graph()
g2 = tf.Graph()
with g1.as_default():
    x = tf.add(3, 5)
with g2.as_default():
    y = tf.add(3, 5)
with tf.Session() as sess:
    print(sess.run(x))