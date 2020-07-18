# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import os
import subprocess
 
# Plant
import warnings
warnings.filterwarnings('ignore') # suppress import warnings

import os
import sys
import cv2
import tflearn
import numpy as np
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))
FACE="NULL"
 


''' <global actions> '''

IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = './model/plant_disease.model'.format(LR, '2conv-basic')
tf.logging.set_verbosity(tf.logging.ERROR) # suppress keep_dims warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow gpu logs

''' </global actions> '''


def analysis(verify_data):

	# verify_data = process_verify_data(filepath)

	str_label = "Cannot make a prediction."
	status = "Error"

	tf.reset_default_graph()

	convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
	
	convnet = conv_2d(convnet, 32, 3, activation='relu')
	convnet = max_pool_2d(convnet, 3)

	convnet = conv_2d(convnet, 64, 3, activation='relu')
	convnet = max_pool_2d(convnet, 3)

	convnet = conv_2d(convnet, 128, 3, activation='relu')
	convnet = max_pool_2d(convnet, 3)

	convnet = conv_2d(convnet, 32, 3, activation='relu')
	convnet = max_pool_2d(convnet, 3)

	convnet = conv_2d(convnet, 64, 3, activation='relu')
	convnet = max_pool_2d(convnet, 3)

	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)
	
	convnet = fully_connected(convnet, 4, activation='softmax')
	convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(convnet, tensorboard_dir='log')

	if os.path.exists('{}.meta'.format(MODEL_NAME)):
		model.load(MODEL_NAME)
		print('Model loaded successfully.')
	else:
		print('Model loaded successfully')

	img_data, img_name = verify_data[0], verify_data[1]

	orig = img_data
	data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)

	model_out = model.predict([data])[0]

	cure = ' '
	contact = '88606107XX'
        
	if np.argmax(model_out) == 0:
                str_label = 'Healthy'
                cure = ''
	elif np.argmax(model_out) == 1:
                str_label = 'Bacterial'
                cure = 'Streptomycin and/or oxytetracycline'
	elif np.argmax(model_out) == 2:
                str_label = 'Viral'
                cure = 'Biocontrol products'
	elif np.argmax(model_out) == 3:
                str_label = 'Lateblight'
                cure = 'Copper based fungicide'

	if str_label =='Healthy': status = 'Healthy'
	else: status = 'Unhealthy'

	

	result = '' + status + '.'
	
	if (str_label != 'Healthy'): result += ', Disease: ' + str_label + ', Cure: ' + cure + ', For details, contact - '+contact+'.'

	return result


@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	data = {}
 
	# check to see if this is a post request
	if request.method == "POST":
		# check to see if an image was uploaded
		if request.FILES.get("image", None) is not None:
			# grab the uploaded image
			image = _grab_image(stream=request.FILES["image"])

			verifying_data = []
			img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
			verifying_data = [np.array(img), ""]
			
			# np.save('data/verify_data.npy', verifying_data)
			# result = subprocess.check_output(['/home/ubuntu/testfr.sh', 'image'])
			result = analysis(verifying_data)
			global FACE
			FACE = result
 
		# otherwise, assume that a URL was passed in
		else:
			# grab the URL from the request
			url = request.POST.get("url", None)
 
			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)
 
			# load the image and convert
			image = _grab_image(url=url)
 
		# update the data dictionary with the faces detected
		data.update({"Plant": FACE})
 
	# return a JSON response
	return JsonResponse(data)
 
def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)
 
	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()
 
		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()
 
		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image
