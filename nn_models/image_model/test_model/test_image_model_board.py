#%%
# This script tests the image model on new unseen data, where the data can be found in following location: 
# /home/udhansen/nn_models/image_model/test_model/input_data
# This is meant to run on the board, since the script uses tflite_runtime.interpreter
#
# Author: Ulrik, s195091
#%%

# Libraries
import tflite_runtime.interpreter as tflite
import numpy as np
import sys
import time

# Image processing libraries
from PIL import Image

TFLITE_FILE_PATH = '/usr/bin/tensorflow-lite-2.6.0/examples/image_model/clothing_classifier_model.tflite'

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Allocate memory for tensors
interpreter = tflite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get info on input and output
# print(input_details)
# print(output_details)

# Allow the user to input the filename of the input image through the terminal
IMAGE_PATH = '/usr/bin/tensorflow-lite-2.6.0/examples/image_model/input_data/' + sys.argv[1]

# Input image processing
# Load input image in grayscale mode
img_loc = Image.open(IMAGE_PATH).convert('L')

# Scale to 28x28
img_loc = img_loc.resize((28, 28))

# Convert it to an array and divide pixel values from 0-255 into range [0, 1]
img_array_loc = np.array(img_loc, dtype = np.float32)
img_array_loc = img_array_loc / 255.0

# Add new dimension
img_array_loc = np.expand_dims(img_array_loc, axis=0)
img_array_loc = -img_array_loc + 1

# Check that the image has been processed in correct way
print('\nInput image information:', img_array_loc.shape, img_array_loc.dtype)
print('\nLoading...\n')

# Get inference time / latency
start_time = time.time()

# Feed the model
interpreter.set_tensor(input_details[0]['index'], img_array_loc)

# Inference
interpreter.invoke()

# Get model output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("\nOutput array: ", output_data)
print("\nPredicted label: ", np.argmax(output_data))
print("Predicted class: ", class_names[np.argmax(output_data)])

# Get inference time
end_time = time.time()
inference_time = end_time - start_time

print("Inference time: {:.4f} seconds\n".format(inference_time))