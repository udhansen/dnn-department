#%%
# This script tests the image model on new unseen data, where the data can be found in following location: 
# /home/udhansen/nn_models/image_model/test_model/input_data
# This is meant to run on the host pc
#
# Author: Ulrik, s195091

#%%
# Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# Image processing libraries
from keras.preprocessing import image
from PIL import Image

# Load tflite model
TFLITE_FILE_PATH = '/home/udhansen/nn_models/image_model/models/tflite/clothing_classifier_model.tflite'

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get info on input and output
print(input_details)
print(output_details)

IMAGE_PATH = '/home/udhansen/nn_models/image_model/test_model/input_data/' + sys.argv[1]

# Input image - LOCAL
img_loc = Image.open(IMAGE_PATH).convert('L')
img_loc = tf.keras.utils.load_img(IMAGE_PATH, color_mode="grayscale")
img_loc = img_loc.resize((28, 28))
img_array_loc = tf.keras.utils.img_to_array(img_loc, data_format="channels_first",dtype=np.float32)
img_array_loc = np.array(img_loc, dtype = np.float32)
img_array_loc = img_array_loc / 255.0
img_array_loc = np.expand_dims(img_array_loc, axis=0)
img_array_loc = -img_array_loc + 1

########################## Imported from 'deploymodelKERAS.py

# Input image - KERAS
# Load the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalizing the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Input image
img_keras = test_images[888]
img_array_keras = tf.keras.utils.img_to_array(img_keras, data_format="channels_first",dtype=np.float32)

##########################

# Comparison on image from mnist fashion image and own image - check if they are in same format. 
# Important that they match otherwise the model can't take any input!
img_comp = [img_keras, img_loc]
img_comp_array = [img_array_keras, img_array_loc]
x_labels = ['keras (from mnist database)', 'locally saved image (attached from web)']

plt.figure(figsize=(8,8))
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.grid(False)
    plt.imshow(img_comp_array[i][0], cmap = plt.cm.binary, vmin=0, vmax=1)
    plt.xlabel(x_labels[i])
plt.show()

# Get inference time / latency
start_time = time.time()

# Feed the model with chosen image
interpreter.set_tensor(input_details[0]['index'], img_array_loc)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
scores_lite = tf.nn.softmax(output_data[0])
print("\nOutput array: ", output_data)
print("\nOutput array (softmax): ", scores_lite)
print("\nPredicted label: ", np.argmax(output_data))
print("Predicted class: ", class_names[np.argmax(output_data)])

# Get inference time
end_time = time.time()
get_inference_time = end_time - start_time

print("Inference time: {:.4f} seconds\n".format(get_inference_time))
