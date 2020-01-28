import numpy as np
import tensorflow as tf

def horizontal_flip (image):
	return tf.image.random_flip_left_right (image)

def random_crop (image, dimensions):
	return tf.image.random_crop (image, dimensions)
