import numpy as np
import tensorflow as tf

from six.moves import cPickle as pickle

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

with open(pickle_file , 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save
	print 'Training set' , train_dataset.shape , train_labels.shape
	print 'Validation set' , valid_dataset.shape , valid_labels.shape
	print 'Test set' , test_dataset.shape , test_labels.shape


# Reformat into a shape that's more adapted to the models we're going to train
image_size = 28
num_labels = 10
def reformat(dataset , labels):
	dataset = dataset.reshape((-1 , image_size * image_size)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[: , None]).astype(np.float32)

	return dataset , labels

train_dataset , train_labels = reformat(train_dataset, train_labels)
valid_dataset , valid_labels = reformat(valid_dataset , valid_labels)
test_dataset , test_labels = reformat(test_dataset , test_labels)

print 'Training set' , train_dataset.shape , train_labels.shape
print 'Validation set' , valid_dataset.shape , valid_labels.shape
print 'Test set' , test_dataset.shape , test_labels.shape

def accuracy(predictions , labels):
	return (100.0 * np.sum(np.argmax(predictions , 1) == np.argmax(labels , 1)) / predictions.shape[0])


train_subset = 100000
graph = tf.Graph()
with graph.as_default():
	# Input data
	tf_train_dataset = tf.constant(train_dataset[:train_subset , :])
	tf_train_labels = tf.constant(train_labels[:train_subset])
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	#Variables
	weights = tf.Variable(tf.truncated_normal([image_size * image_size , num_labels]))
	biases = tf.Variable(tf.zeros([num_labels]))

	#activation_function  and  loss
	logits = tf.matmul(tf_train_dataset , weights) + biases
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits , tf_train_labels))
	#regular = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)

	#loss += regular

	#Optimizer
	Optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	# Prediction
	train_pre = tf.nn.softmax(logits)
	valid_pre = tf.nn.softmax(tf.matmul(tf_valid_dataset , weights) + biases)
	test_pre = tf.nn.softmax(tf.matmul(tf_test_dataset , weights) + biases)


num_steps = 801

with tf.Session(graph = graph) as session:
	tf.initialize_all_variables().run()
	print 'intialized'

	for step in range(num_steps):
		_ , l , predictions = session.run([Optimizer , loss , train_pre])

		if step % 100 == 0:
			print 'Loss at step %d : %f' % (step , l)
			print 'Training accuracy: %.1f%%' % accuracy(predictions , train_dataset[:train_subset , :])

			print 'Validation accuracy: %.1f%%' % accuracy(valid_pre.eval() , valid_labels)

	print 'Test accuracy: %.1f%%' % accuracy(test_pre.eval() , test_labels)




