# Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Image processing libraries
from keras.preprocessing import image
from PIL import Image

# Load the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalizing the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Input image
img = test_images[888]
img_array = tf.keras.utils.img_to_array(img, data_format="channels_first",dtype=np.float32)

TFLITE_FILE_PATH = '/home/udhansen/nn_models/image_model/models/tflite/clothing_classifier_model.tflite'

interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

plt.figure(1)
plt.imshow(img)
plt.show()

print(img_array.shape)

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
scores_lite = tf.nn.softmax(output_data[0])
print("Scores: ", scores_lite)
print("\nOutput array: ", output_data)
print("\nOutput details: ", scores_lite)
print("\nOutput class label =", np.argmax(scores_lite))
