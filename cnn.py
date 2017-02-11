#coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data' , one_hot=True)

def compute_accuracy(v_xs , v_ys):
	global prediction
	y_pre = sess.run(prediction , feed_dict = {xs:v_xs , ys:v_ys , keep_prob:1})
	correct_prediction = tf.equal(tf.argmax(y_pre , 1) , tf.argmax(v_ys , 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
	result = sess.run(accuracy , feed_dict = {xs:v_xs , ys:v_ys	 , keep_prob: 1})

	return result


def weight_variable(shape):
	initial = tf.truncated_normal(shape , stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1 , shape = shape)
	return tf.Variable(initial)

def conv2d(x , W):
	#stride[1 , x_move , y_move , 1]
	#padding  零填充
	return tf.nn.conv2d(x , W , strides = [1 , 1 , 1 , 1] , padding='SAME')

# 为了防止跨度过大，信息丢失
def max_pool_2x2(x):
	return tf.nn.max_pool(x , ksize = [1 , 2 , 2 , 1] , strides = [1 , 2, 2 , 1] , padding='SAME') #ksize : The size of the window for each dimension of the input tensor.

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32 , [None , 784] , name='x_input')
    ys = tf.placeholder(tf.float32 , [None , 10] , name='y_input')
    keep_prob = tf.placeholder(tf.float32 , name = 'Dropout_parameter')
x_image = tf.reshape(xs , [-1 , 28 , 28 , 1])
# print(x_image.shape)  # [n_samples , 28 , 28 , 1(维度)]

#conv 1 layer
with tf.name_scope('conv_layer_1'):
	with tf.name_scope('weigits'):
		W_conv1 = weight_variable([5 ,5 , 1 , 32]) # patch 5x5 , in_size = 1, out_size = 32
	with tf.name_scope('biases'):
		b_conv1 = bias_variable([32])
	with tf.name_scope('activation_function'):
		h_conv1 = tf.nn.relu(conv2d(x_image , W_conv1) + b_conv1) # outpus size 14x14x32 (padding = SAME)

with tf.name_scope('pooling'):
	h_pool1 = max_pool_2x2(h_conv1) # output size 7 x7 x 32 strides[1 ,2 , 2 , 1]


# conv2 layer
with tf.name_scope('conv_layer_2'):
	with tf.name_scope('weights'):
		W_conv2 = weight_variable([5 ,5 , 32 , 64]) # patch 5x5 , in_size = 32, out_size = 64
	with tf.name_scope('biases'):
		b_conv2 = bias_variable([64])
	with tf.name_scope('activation_function'):
		h_conv2 = tf.nn.relu(conv2d(h_pool1 , W_conv2) + b_conv2) # outpus size 28x28x64 (padding = SAME)
with tf.name_scope('pooling'):
	h_pool2 = max_pool_2x2(h_conv2) # output size 14 x14 x 32 strides[1 ,2 , 2 , 1]

# func1 layer
with tf.name_scope('layer3'):
	with tf.name_scope('W_fc1'):
		W_fc1 = weight_variable([7*7*64 , 1024])
	with tf.name_scope('b_fc1'):
		b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2 , [-1 , 7*7*64]) #[n_samples , 7 , 7 , 64] ->> [n , samples , 7*7*64]
	with tf.name_scope('activation_func'):
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat , W_fc1) + b_fc1)
	with tf.name_scope('Dropout'):
		h_fc1_drop = tf.nn.dropout(h_fc1 , keep_prob)


# func 2 layer
with tf.name_scope('layer4'):
	with tf.name_scope('W_fc2'):
		W_fc2 = weight_variable([1024 , 10])
	with tf.name_scope('b_fc2'):
		b_fc2 = bias_variable([10])

	with tf.name_scope('activation_func'):
		prediction = tf.nn.softmax(tf.matmul(h_fc1_drop , W_fc2) + b_fc2)


# the error between prediction and real data
with tf.name_scope('loss'):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction) , reduction_indices = [1]))

with tf.name_scope('Train'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
merged = tf.merge_all_summaries()

writer = tf.train.SummaryWriter("logs/" , sess.graph)
sess.run(tf.initialize_all_variables())


for i in range(100):
	batch_xs , batch_ys = mnist.train.next_batch(1000)
	sess.run(train_step , feed_dict = {xs: batch_xs , ys:batch_ys , keep_prob: 0.5})
	if i % 50 == 0:


		print compute_accuracy(mnist.test.images , mnist.test.labels)

