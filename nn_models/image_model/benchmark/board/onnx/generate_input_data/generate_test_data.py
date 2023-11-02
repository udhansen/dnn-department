#
# This script generates .pb input files for ONNX Runtime profiling. The generated .pb file can be checked with following script: 
# /home/udhansen/nn_models/image_model/benchmark/board/onnx/generate_input_data/check_pb_file.py
#
# Author: Ulrik, s195091
#%% 
import numpy as np
import onnxruntime as ort
from onnx import numpy_helper

#%% 
# Load the ONNX model
ONNX_FILE_PATH = '/home/udhansen/nn_models/image_model/models/onnx/clothing_classifier_model.onnx'
sess = ort.InferenceSession(ONNX_FILE_PATH)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
input_shape = sess.get_inputs()[0].shape
output_shape = sess.get_outputs()[0].shape

# Get input and output details
print(input_name)
print(input_shape)
print(output_name)
print(output_shape)

#%% 
# Function to generate random test data
def generate_test_data(input_shape):
    return np.random.rand(*input_shape).astype(np.float32)

# Input shape
IMAGE_SHAPE = (1, 28, 28)

# Generate test data for benchmark
test_data = generate_test_data(IMAGE_SHAPE)
#%% 
# Convert from numpy array into binary file
tensor_proto = numpy_helper.from_array(test_data)
tensor_proto.name=input_name
#%%
# Generate .pb file 
with open('input_0.pb', 'wb') as f:
    f.write(tensor_proto.SerializeToString())
# %%
