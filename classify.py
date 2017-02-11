import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data' , one_hot = True)

def add_layer(inputs , in_size , out_size , activation_func = None):
	Weights = tf.Variable(tf.random_normal([in_size , out_size]))
	biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1 , )

	Wx_Plus_b = tf.add(tf.matmul(inputs , Weights) , biases)

	if activation_func is None:
		ot = Wx_Plus_b
	else:
		ot = activation_func(Wx_Plus_b , )

	return ot


def compute_accuracy(v_xs , v_ys):
	global prediction
	y_pre = sess.run(prediction , feed_dict={xs : v_xs})
	correct_prediction = tf.equal(tf.argmax(y_pre , 1) , tf.argmax(v_ys , 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
	result = sess.run(accuracy , feed_dict = {xs: v_xs , ys: v_ys})
	return result


xs = tf.placeholder(tf.float32 , [None , 784]) # 28x28
ys = tf.placeholder(tf.float32 , [None , 10])


prediction = add_layer(xs , 784 , 10 , activation_func = tf.nn.softmax)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction) , reduction_indices=[1])) #loss function

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess =  tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
	batch_xs , batch_ys = mnist.train.next_batch(100)
	sess.run(train_step , feed_dict={xs:batch_xs , ys:batch_ys})
	if i % 50 == 0:
		print compute_accuracy(mnist.test.images , mnist.test.labels)
