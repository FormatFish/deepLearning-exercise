#coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def add_layer(inputs , in_size , out_size , n_layer,  activation_function=None):
	layer_name = 'layer%s' % n_layer
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):		
			Weights = tf.Variable(tf.random_normal([in_size , out_size]) , name='W')
			tf.histogram_summary(layer_name+'/weights' , Weights)

		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1 , name='b')
			tf.histogram_summary(layer_name+'/biases' , biases)

		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs ,Weights) , biases)

		if activation_function is None:
			ot = Wx_plus_b
		else:
			ot = activation_function(Wx_plus_b,)
		tf.histogram_summary(layer_name+'/output' , ot)


		return ot

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0 , 0.05 , x_data.shape)
y_data = np.square(x_data) - 0.5 +noise#训练数据 y = x^2


with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32 , [None , 1] , name='x_input')
	ys = tf.placeholder(tf.float32 , [None , 1] , name='y_inpuut')
## hidden layer
l1 = add_layer(xs , 1 , 10 , n_layer = 1 , activation_function = tf.nn.relu)

## outpur layer
predition = add_layer(l1 , 10 , 1 , n_layer = 2 , activation_function =None)

with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition) ,  reduction_indices =[1])) #reduction_indices代表取平均值的维度
	tf.scalar_summary('loss' , loss)

with tf.name_scope('Train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)#优化损失函数（梯度下降法）

init = tf.initialize_all_variables()

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("logs/" , sess.graph)
sess.run(init)

'''
fig = plt.figure()
ax = fig.add_subplot(1 , 1 , 1)
ax.scatter(x_data , y_data)
#plt.ion()
plt.show(block = False)
'''

for i in range(1000):
	sess.run(train_step , feed_dict = {xs :x_data , ys: y_data})

	if i % 50 == 0:
		#print sess.run(loss , feed_dict = {xs:x_data , ys:y_data})
		result = sess.run(merged , feed_dict={xs:x_data , ys:y_data})
		writer.add_summary(result , i)


