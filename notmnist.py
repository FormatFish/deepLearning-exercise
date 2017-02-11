#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf

from IPython.display import display , Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from six.moves import range

#%matplotlib inline

#download datasets

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

'''
@count 已经下载的数据块
@blockSize 数据块的大小
@totalSize 远程文件的大小
'''
def download_progress_hook(count , blockSize , totalSize): #report the progress in the terminal
	global last_percent_reported
	percent = int(count * blockSize * 100 / totalSize)

	if last_percent_reported != percent:
		if percent % 5 == 0:
		    sys.stdout.write("%s%%" % percent)
		    sys.stdout.flush()
		else:
			sys.stdout.write(".")
			sys.stdout.flush()

		last_percent_reported = percent


def maybe_donwload(filename , expected_bytes , force = False):
	if force or not os.path.exists(filename):
		print 'Attempting to download: ' , filename
		filename , _ = urlretrieve(url + filename , filename , reporthook = download_progress_hook)
		print '\nDownload Complete!'

	statinfo = os.stat(filename)

	if statinfo.st_size == expected_bytes:
		print 'Found and verified' , filename
	else:
		raise Exception('Failed to verify' + filename+'.Can you get to it with a browser?')

	return filename

train_filename = maybe_donwload('notMNIST_large.tar.gz' , 247336696)
test_filename = maybe_donwload('notMNIST_small.tar.gz' , 8458043)


# extract the dataset from the compressed .tar.gz file. This should give you a set of directories , labelled A through J.

num_classes = 10
np.random.seed(133)
def maybe_extract(filename , force = False):
	root = os.path.splitext(os.path.splitext(filename)[0])[0] # remove .tar.gz
	if os.path.isdir(root) and not force:
		print '%s already present - Skipping extraction of %s.' % (root , filename)
	else:
		print 'Extracting data for %s. this may take a while. Please wait. ' % root
		tar = tarfile.open(filename)
		sys.stdout.flush()
		tar.extractall()
		tar.close()
	data_folders = [os.path.join(root , d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root , d))]
	if len(data_folders) != num_classes:
		raise Exception('Expected %d folders , one per class. Found %d instead. ' % (num_classes , len(data_folders)))

	print data_folders
	return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

# covert images data to sensible
image_size = 28 # Pixel width and height
pixel_depth = 255.0 # Number of levels per pixel

def load_letter(folder , min_num_images):
	image_files = os.listdir(folder)
	dataset = np.ndarray(shape = (len(image_files) , image_size , image_size) , dtype = np.float32)
	print folder

	num_images = 0
	for image in image_files:
		image_file = os.path.join(folder , image)
		try:
			image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth #参数规范均值和方差
			if image_data.shape != (image_size , image_size):
				raise Exception('Unexpected image shape: %s ' % str(image_data.shape))
			dataset[num_images , : , :] = image_data
			num_images = num_images +1
		except IOError as e:
			print 'Could not read:' , image_file , ':' , e , '- it\'s ok , skipping.'

	dataset = dataset[0:num_images , : , :]
	if num_images < min_num_images:
		raise Exception('Many fewer images than expected : %d < %d ' % (num_images , min_num_images))

	print 'Full dataset tensor:' , dataset.shape
	print 'Mean:' , np.mean(dataset)
	print 'standard deviation:' , np.std(dataset)
	return dataset

#每个类对应一个dataset
'''
这里data_folders的样例：
['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E', 'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 'notMNIST_large/I', 'notMNIST_large/J']
['notMNIST_small/A', 'notMNIST_small/B', 'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E', 'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 'notMNIST_small/I', 'notMNIST_small/J']
'''
def maybe_pickle(data_folders , min_num_images_per_class , force = False):
	dataset_names = []
	for folder in data_folders:
		set_filename = folder + '.pickle'
		dataset_names.append(set_filename)
		if os.path.exists(set_filename) and not force:
			print '%s already present - Skipping pickling.' % set_filename
		else:
			print 'Pickling %s.' % set_filename
			dataset = load_letter(folder , min_num_images_per_class)
			try:
				with open(set_filename , 'wb') as f:
					pickle.dump(dataset , f , pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print 'Unable to save data to' , set_filename , ':' , e

	return dataset_names

train_datasets = maybe_pickle(train_folders , 45000)#这里存储的是文件名称
test_datasets = maybe_pickle(test_folders , 1800)

'''
pickel_file = train_datasets[0]
with open(pickel_file , 'rb') as f:
	letter_set = pickle.load(f)
	sample_idx = np.random.randint(len(letter_set))
	sample_image = letter_set[sample_idx , : , :]
	plt.figure()
	plt.imshow(sample_image)
'''
#tune train_size as needed and create a validation dataset for hyperparameter tuning
def make_arrays(nb_rows , img_size):
	if nb_rows:
		dataset = np.ndarray((nb_rows , img_size , img_size) , dtype = np.float32)
		labels = np.ndarray(nb_rows , dtype = np.int32)
	else:
		dataset , labels = None , None
	return dataset , labels

def merge_datasets(pickle_files , train_size , valid_size = 0):
	num_classes = len(pickle_files)
	valid_dataset , valid_labels = make_arrays(valid_size , image_size)
	train_dataset , train_labels = make_arrays(train_size , image_size)
	vsize_per_class = valid_size // num_classes
	tsize_per_class = train_size // num_classes

	start_v , start_t = 0 , 0
	end_v , end_t = vsize_per_class , tsize_per_class
	end_l = vsize_per_class + tsize_per_class

	for label , pickle_file in enumerate(pickle_files):#这里返回的是一个元祖，(idx , coll(idx))
		try:
			with open(pickle_file , 'rb') as f:
				letter_set = pickle.load(f)
				np.random.shuffle(letter_set) #这里将每个分类的数据集打乱，保证随机性
				if valid_dataset is not None:
					valid_letter = letter_set[:vsize_per_class , : , :]
					valid_dataset[start_v:end_v , : , :] = valid_letter
					valid_labels[start_v : end_v] = label
					start_v += vsize_per_class
					end_v += vsize_per_class

				train_letter = letter_set[vsize_per_class:end_l , : , :] #除了验证集那就是训练集咯
				train_dataset[start_t:end_t , : , :] = train_letter
				train_labels[start_t : end_t] = label
				start_t += tsize_per_class
				end_t += tsize_per_class

		except Exception as e:
			print 'Unable to process data form' , pickle_file , ':' , e
			raise
	return valid_dataset , valid_labels , train_dataset , train_labels

train_size = 200000
valid_size = 10000
test_size = 10000
valid_dataset , valid_labels , train_dataset , train_labels = merge_datasets(train_datasets , train_size , valid_size)
_ , _ , test_datasets , test_labels = merge_datasets(test_datasets , test_size)

print 'Training:' , train_dataset.shape , train_labels.shape
print '\nValidation:' , valid_dataset.shape , valid_labels.shape
print '\nTesting:' , test_datasets.shape , test_labels.shape


#randomize the data.
def randomize(dataset , labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation , : , :]
	shuffled_labels = labels[permutation]

	return shuffled_dataset , shuffled_labels

train_dataset , train_labels = randomize(train_dataset , train_labels)
test_datasets , test_labels = randomize(test_datasets , test_labels)
valid_dataset , valid_labels = randomize(valid_dataset , valid_labels)


#save the data for later reuse
pickle_file = 'notMNIST.pickle'
try:
	f = open(pickle_file , 'wb')
	save = {
	    'train_dataset': train_dataset , 
	    'train_labels' : train_labels , 
	    'valid_dataset' : valid_dataset , 
	    'valid_labels' : valid_labels , 
	    'test_dataset' : test_datasets , 
	    'test_labels' : test_labels
	}
	pickle.dump(save , f , pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	print 'Unable to save data to' , pickle_file , ':' , e
	raise

statinfo = os.stat(pickle_file)
print 'Compressed pickle size:' , statinfo.st_size



