import numpy as np
import tensorflow as tf

def random_cutout (image, image_height, image_width, cutout_num, is_mean = False):
	
	# create a random x, y cordinate to take cutout
	rand_x = np.random.randint (0, image_height + cutout_num)
	rand_y = np.random.randint (0, image_width + cutout_num)

	# create a new matrix of ones with increased dimensions
	new_image = np.ones ([image_height + cutout_num, image_width + cutout_num, 3])
	# make the cutout area 0
	new_image[rand_x : rand_x + cutout_num, rand_y : rand_y + cutout_num, :] = 0
	# cut the original center of the matrix
	new_image = new_image[cutout_num : cutout_num + image_height, cutout_num : cutout_num + image_width, :]

	# replace the cutout with mean of the image
	if is_mean == True:
		# find mean along each axis
		mean_value = tf.reduce_mean(image, axis=(0, 1))
		# create a matrix with mean replacing 0 in 0 replacing 1 in new_image matrix
		mean_image = 1 - new_image
		mean_image = mean_image * mean_value

	# convert both matrix to tensor for tensor operations
	if is_mean == True:
		mean_image = tf.convert_to_tensor (mean_image, dtype='float32')
	new_image = tf.convert_to_tensor (new_image, dtype='float32')

	# cutout will be black spot (0) in this operation
	image = image * new_image
	# 0/cutout will be replaced by mean
	if is_mean == True:
		image = image + mean_image
	
	return image
