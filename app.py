# Import required packages:
from flask import Flask, request, render_template, send_from_directory, redirect, send_file
import os

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
	return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
	import tensorflow_hub as hub
	import tensorflow as tf
	
	import numpy as np
	import PIL.Image
	
	###################################
	content_path = tf.keras.utils.get_file('lucinta.jpg', 'https://thumb.viva.co.id/media/frontend/thumbs3/2021/05/03/608fbebb52d12-lucinta-luna_665_374.jpg')
	style_path = tf.keras.utils.get_file('tradi.jpg','https://previews.123rf.com/images/wangdu88/wangdu881908/wangdu88190800282/128837608-flower-and-bird-traditional-chinese-painting.jpg')
	
	#####################################################
	def load_img(path_to_img):
		max_dim = 512
		img = tf.io.read_file(path_to_img)
		img = tf.image.decode_image(img, channels=3)
		img = tf.image.convert_image_dtype(img, tf.float32)
		shape = tf.cast(tf.shape(img)[:-1], tf.float32)
		long_dim = max(shape)
		scale = max_dim / long_dim
		new_shape = tf.cast(shape * scale, tf.int32)
		img = tf.image.resize(img, new_shape)
		img = img[tf.newaxis, :]
		return img
	
	def tensor_to_image(tensor):
		tensor = tensor*255
		tensor = np.array(tensor, dtype=np.uint8)
		if np.ndim(tensor)>3:
			assert tensor.shape[0] == 1
			tensor = tensor[0]
		return PIL.Image.fromarray(tensor)
	
	
	#display the image
	content_image = load_img(content_path)
	style_image = load_img(style_path)
	
	hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
	stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
	tensor_to_image(stylized_image)
	newImage = tensor_to_image(stylized_image)
	
	file_name = 'stylized2'+ '_' + '.jpg'
	
	#save the image
	tensor_to_image(stylized_image).save(file_name)
	
	#from google.colab import files
	#files.download(file_name)
	
	return render_template("result.html", image_name=file_name)

#good practise to have this: this means this will only run if its run directly (and not called from somewhere else)
if __name__ == "__main__":
	#remove debug and host when hosting to cloud
	# Add parameter host='0.0.0.0' to run on your machines IP address:
	app.run(host='0.0.0.0')


