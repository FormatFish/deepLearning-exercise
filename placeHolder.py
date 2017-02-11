import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

ot = tf.mul(input1 , input2)

with tf.Session() as sess:
	print sess.run(ot , feed_dict ={input1: [7.0] , input2: [2.0]})


