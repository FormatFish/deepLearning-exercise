#coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)

mnist = input_data.read_data_sets('MNIST_data' , one_hot=True)

# hyperparameters
lr = 0.001
trainning_iters = 100000
bathc_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

# input
x = tf.placeholder(tf.float32 , [None , n_steps , n_inputs])
y = tf.placeholder(tf.float32 , [None , n_classes])

#define weights and biases
Weights = {
	#(28 , 128)
	'in': tf.Variable(tf.random_normal([n_inputs , n_hidden_units])) , 

	#(128 , 10)
	'out': tf.Variable(tf.random_normal([n_hidden_units , n_classes]))
}

biases = {
	#(128 , )
	'in': tf.Variable(tf.constant(0.1 , shape = [n_hidden_units , ])) , 

	#(10 , )
	'out': tf.Variable(tf.constant(0.1 , shape = [n_classes , ]))
}


# RNN nn layer
def RNN(X , Weights , biases):
	#hidden layer for input to cell
	###########################################
	#X(128 batch, 28 step , 28 inputs pixels)
	# => (128 * 28 , 28 inputs)
	X = tf.reshape(X , [-1  , n_inputs])
	# X_in=>(128 batch * 28 steps , 128 hidden)
	X_in = tf.matmul(X , Weights['in']) + biases['in']
	# X_in=>(128 batch  , 28 steps , 128 hidden)
	X_in = tf.reshape(X_in , [-1 , n_steps , n_hidden_units])


	#cell
	#######################
	lstm_cell =tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units , forget_bias = 1.0 , state_is_tuple = True)
	# lstm cell si divided into two parts (c_state , m_state)(主线状态，分线状态)
	_init_state = lstm_cell.zero_state(bathc_size , dtype = tf.float32)

	outputs , states = tf.nn.dynamic_rnn(lstm_cell , X_in , initial_state = _init_state , time_major = False)
	

	# hidden layer for output from cell
	########################################
	result = tf.matmul(states[1] , Weights['out']) + biases['out']# states[1]就是最后一个output的结果

	## method 2
	# unpack to list [(batch , outputs)...] * steps
	#outputs = tf.unpack(tf.transpose(outputs , [1 , 0 , 2]))
	#result = tf.matmul(outputs[-1] , Weights['out']) + biases['out']

	return result

pred = RNN(x , Weights , biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred , y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred , 1) , tf.argmax(y , 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred , tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	step = 0
	while step * bathc_size < trainning_iters:
		batch_xs , batch_ys = mnist.train.next_batch(bathc_size)
		batch_xs = batch_xs.reshape([bathc_size , n_steps , n_inputs])
		sess.run([train_op] , feed_dict = {x: batch_xs , y: batch_ys})

		if step % 20 == 0:
			print sess.run(accuracy , feed_dict = {x:batch_xs , y:batch_ys})

		step += 1
