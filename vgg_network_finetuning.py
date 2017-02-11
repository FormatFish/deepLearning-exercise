#coding=utf-8
'''
fine tune
'''

import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os

def vgg16(input , num_class):
	x = tflearn.conv_2d(input , 64 , 3 , activation = 'relu' , scope = 'conv1_1')
	x = tflearn.conv_2d(x , 64 , 3 , activation = 'relu' , scope = 'conv1_2')
	x = tflearn.max_pool_2d(x , 2 , strides = 2 , name = 'maxpool1')

	x = tflearn.conv_2d(x , 128 , 3 , activation = 'relu' , scope = 'conv2_1')
	x = tflearn.conv_2d(x , 128 , 3 , activation = 'relu' , scope = 'conv2_2')
	x = tflearn.max_pool_2d(x , 2 , strides = 2 , name = 'maxpool2')

	x = tflearn.conv_2d(x , 256 , 3 , activation = 'relu' , scope = 'conv3_1')
	x = tflearn.conv_2d(x , 256 , 3 , activation = 'relu' , scope = 'conv3_2')
	x = tflearn.conv_2d(x , 256 , 3 , activation = 'relu' , scope = 'conv3_3')
	x = tflearn.max_pool_2d(x , 2 , strides = 2 , name = 'maxpool3')

	x = tflearn.conv_2d(x , 512 , 3 , activation = 'relu' , scope = 'conv4_1')
	x = tflearn.conv_2d(x , 512 , 3 , activation = 'relu' , scope = 'conv4_2')
	x = tflearn.conv_2d(x , 512 , 3 , activation = 'relu' , scope = 'conv4_3')
	x = tflearn.max_pool_2d(x , 2 , strides = 2 , name = 'maxpool4')

	x = tflearn.conv_2d(x , 512 , 3 , activation = 'relu' , scope = 'conv5_1')
	x = tflearn.conv_2d(x , 512 , 3 , activation = 'relu' , scope = 'conv5_2')
	x = tflearn.conv_2d(x , 512 , 3 , activation = 'relu' , scope = 'conv5_3')
	x = tflearn.max_pool_2d(x , 2 , strides = 2 , name = 'maxpool5')

	x = tflearn.fully_connected(x , 4096 , activation = 'relu' , scope = 'fc6')
	x = tflearn.dropout(x , 0.5 , name = 'dropout1')

	x = tflearn.fully_connected(x , 4096 , activation = 'relu' , scope = 'fc7')
	x = tflearn.dropout(x , 0.5 , name = 'dropout2')

	x = tflearn.fully_connected(x , num_class , activation = 'softmax' , scope = 'fc8' , restore = False)

	return x


# fine tune 精髓就是用已有的训练好的模型来对只有少量样本的数据进行微调，然后使用
data_dir = '/path/to/your/data'
model_path = '/path/to/your/vgg_modell'

files_list = '/path/to/your/file/with/images'
'''
/path/to/img1 class_id
/path/to/img2 class_id
/path/to/img3 class_id

files_list存放路径和类别，data_dir存放着对应图片
'''
from tflearn.data_utils import image_preloader

X , Y = image_preloader(files_list , image_shape = (224 , 224) , mode = 'file' , categorical_labels = True , normalize = False , files_extension = ['.jpg' , '.png'] , filter_channel = True)
# or use the mode 'floder'
# X, Y = image_preloader(data_dir, image_shape=(224, 224), mode='folder',
#                        categorical_labels=True, normalize=True,
#                        files_extension=['.jpg', '.png'], filter_channel=True)
'''
这里如果mode = folder , 那么结构应该是这样子的
ROOT_FOLDER -> SUBFOLDER_0 (CLASS 0) -> CLASS0_IMG1.jpg -> CLASS0_IMG2.jpg -> ...-> SUBFOLDER_1 (CLASS 1) -> CLASS1_IMG1.jpg -> ...-> ...
'''
num_class = 10 # num of your dataset

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(mean = [123.68 , 116.779 , 103.939] , per_channel = True) # 因为图片有3个通道

x = tflearn.input_data(shape = [None , 224 , 224 , 3] , name = 'input' , data_preprocessing = img_prep)

softmax = vgg16(x , num_class)
regression = tflearn.regression(softmax , optimizer = 'adam' , loss = 'categorical_crossentropy' , learning_rate = 0.001 , restore = False)

model = tflearn.DNN(regression , checkpoint_path = 'vgg_finetuning' , max_checkpoints = 3 , tensorboard_verbose = 2 , tensorboard_dir = './logs')

#加载已有的模型文件
model_file = os.path.join(model_path , 'vgg16.tflearn')
model.load(model_file , weight_only = True)

#Start finetuning
model.fit(X , Y , n_epoch = 10 , validation_set = 0.1 , shuffle = True , show_metric = True , batch_size = 64 , snapshot_epoch = False , snapshot_step = 200  , run_id = 'vgg-finetuning')

model.save('your-task-model-retrained-by-vgg')
