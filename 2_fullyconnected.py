#coding=utf-8
###############Assignments 2 train deeper and more accurate models using Tensorflow ############

import numpy as np
import tensorflow as tf

from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

'''
save = {
	    'train_dataset': train_dataset , 
	    'train_labels' : train_labels , 
	    'valid_dataset' : valid_dataset , 
	    'valid_labels' : valid_labels , 
	    'test_dataset' : test_datasets , 
	    'test_labels' : test_labels
	}
'''
with open(pickle_file) as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save # gc
	print 'Training set' , train_dataset.shape , train_labels.shape
	print 'Validation set' , valid_dataset.shape , valid_labels.shape
	print 'Test set' , test_dataset.shape , test_labels.shape


# Reformat into a shape that's more adapted to the models we're going totrain
image_size = 28
num_labels = 10
def reformat(dataset , labels):
	dataset = dataset.reshape((-1 , image_size * image_size)).astype(np.float32)
	# One-Hot encoding
	labels = (np.arange(num_labels) == labels[: , None]).astype(np.float32)#这里因为label其实在前面设置的时候就是0-9的数字，所以这里的手法很高明
	return dataset , labels

train_dataset , train_labels = reformat(train_dataset , train_labels)
valid_dataset , valid_labels = reformat(valid_dataset , valid_labels)
test_dataset , test_labels = reformat(test_dataset , test_labels)

print 'Training set' , train_dataset.shape , train_labels.shape
print 'Validation set' , valid_dataset.shape , valid_labels.shape
print 'Test set' , test_dataset.shape , test_labels.shape

'''
#### train a multinormal logistic regression using simple gradient descent
train_subset = 10000

graph = tf.Graph()
with graph.as_default():
	# Input data
	# Load the training , validation , and test data into constant that are attached to the graph
	tf_train_dataset = tf.constant(train_dataset[:train_subset , :])
	tf_train_labels = tf.constant(train_labels[:train_subset])
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	# Variables (hyperparameter)
	# These are the parameters that we are going to be training.
	# The weights matrix will be initialized using random values for following a (truncated) normal distribution.
	# The biases get initialized to zero
	weights = tf.Variable(tf.truncated_normal([image_size * image_size , num_labels])) # (28*28 , 10)
	biases = tf.Variable(tf.zeros([num_labels]))

	# Training computation
	#Wx+ b and compute the softmax and cross-entropy(loss function). We take the average of this cross-entropy across
	# all training examples: that's our loss
	logits = tf.matmul(tf_train_dataset , weights) + biases

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits , tf_train_labels)) #pre and train_labels loss

	# Optimizer
	# We are going to find the minimum of this loss using gradient descent
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	# Predictions for the training , validation , and test data
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset , weights) + biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset , weights) + biases)

#run this computation and iterate
num_steps = 801
def accuracy(predictions , labels):
	return (100.0 * np.sum(np.argmax(predictions , 1) == np.argmax(labels , 1)) / predictions.shape[0])

with tf.Session(graph = graph) as session:
	# This is a one-time operation which ensures the parameters get initialized as
	# we described in the graph : random weights for the matrix , zeros for the biases
	tf.initialize_all_variables().run()
	print 'Initialized'

	for step in range(num_steps):
		# Run the computations.
		# We tell .run() that we want to tun the optimizer , 
		# and get the loss value and the training predictions returned as numpy arrays
		_ , l , predictions = session.run([optimizer , loss , train_prediction]) #每一个参数对应一个输出
		if step % 100 == 0 :
			print 'Loss at step %d :  %f' % (step , l)
			print 'Training accuracy: %.1f%%' % accuracy(predictions , train_labels[:train_subset , :])

			# Calling .eval() on valid_prediction is basically like calling run() , but
			# just to get that one numpy array. Note that it recomputes all its graph dependencies
			print 'Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval() , valid_labels)

	print 'Test accuracy: %.1f%%' % accuracy(test_prediction.eval() , test_labels)
'''

# switch to stochastic gradient descent training instead
# The graph will be similar, except that instead of holding all the training data into a constant node
# We create a PLaceholder node which will be fed actual data at every call of session.run()
batch_size = 128

def accuracy(predictions , labels):
	return (100.0 * np.sum(np.argmax(predictions , 1) == np.argmax(labels , 1)) / predictions.shape[0])

graph = tf.Graph()
with graph.as_default():
	#Input data
	tf_train_dataset = tf.placeholder(tf.float32 , shape = (batch_size , image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32 , shape = (batch_size , num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	#Variables.
	weights = tf.Variable(tf.truncated_normal([image_size * image_size , num_labels]))
	biases = tf.Variable(tf.zeros([num_labels]))

	# Training computation
	logits = tf.matmul(tf_train_dataset , weights) + biases
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits , tf_train_labels))

	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	# Predictions for the training , validation , and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset , weights) + biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset , weights) + biases)

## run it
num_steps = 3001

with tf.Session(graph = graph) as session :
	tf.initialize_all_variables().run()
	print 'Initialized'

	for step in range(num_steps):
		#Pick an offser within the training data , which has been randomized.
		# Note: we could use better randomization across epochs
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

		# Generate a minibatch
		batch_data = train_dataset[offset: (offset + batch_size) , :]
		batch_labels = train_labels[offset : (offset + batch_size) , :]

		# Prepare a dict telling the session where to feed the minibatch
		feed_dict = {tf_train_dataset : batch_data , tf_train_labels : batch_labels}

		_ , l , predictions = session.run([optimizer , loss ,  train_prediction] , feed_dict = feed_dict)

		if step % 500 == 0 :
			print 'Minibatch loss at step %d: %f' % (step , l)
			print 'Minibatch accuracy: %.1f%%' % accuracy(predictions , batch_labels)
			print 'Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval() , valid_labels)

	print 'Test accuracy: %.1f%%' % accuracy(test_prediction.eval() , test_labels)