#coding=utf-8
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = .3)


def add_layer(inputs , in_size , out_size , layer_name , activation_function = None , ):
	Weight = tf.Variable(tf.random_normal([in_size , out_size]))
	biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1 , )
	Wx_plus_b = tf.matmul(inputs , Weight) + biases
	Wx_plus_b = tf.nn.dropout(Wx_plus_b , keep_prob) #dropout 掉keep_prob的结果
	if activation_function is None :
		ot = Wx_plus_b
	else:
		ot = activation_function(Wx_plus_b,)
	tf.histogram_summary(layer_name + '/outputs' , ot)

	return ot

#define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32) #一直保持多少的结果不被dropout
xs = tf.placeholder(tf.float32 , [None , 64]) #8x8
ys = tf.placeholder(tf.float32 , [None , 10])

# add out_layer
l1 = add_layer(xs , 64 , 50 , 'l1' , activation_function=tf.nn.tanh) #这里为什么必须用tanh？处理信息会变成None
																	# 如果出现莫名其妙的错误，应该是这一部分的问题，out_size选的过大
prediction = add_layer(l1 , 50 , 10 , 'l2' , activation_function=tf.nn.softmax) 

# the loss between prediction and real data
corss_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction) , reduction_indices=[1]))

tf.scalar_summary('loss' , corss_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(corss_entropy)

sess = tf.Session()
merged = tf.merge_all_summaries()
# summary writer goes in here
train_writer = tf.train.SummaryWriter("logs/train" , sess.graph)
test_writer = tf.train.SummaryWriter("logs/test" , sess.graph)

init = tf.initialize_all_variables()
sess.run(init)

for i in range(500):
	sess.run(train_step , feed_dict={xs: X_train , ys:y_train , keep_prob: 0.5})
	if i % 50 == 0:
		train_result = sess.run(merged , feed_dict={xs: X_train , ys:y_train , keep_prob: 1})
		test_result = sess.run(merged , feed_dict ={xs: X_train , ys:y_train , keep_prob: 1})

		train_writer.add_summary(train_result , i)
		test_writer.add_summary(test_result , i)
