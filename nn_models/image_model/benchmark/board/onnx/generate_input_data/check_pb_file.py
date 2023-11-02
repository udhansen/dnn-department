#%% 
#
# This script checks the generated .pb file for ONNX Runtime profiling. The generated .pb file is generated from following script: 
# /home/udhansen/nn_models/image_model/benchmark/board/onnx/generate_input_data/generate_test_data.py
#
# Author: Ulrik, s195091
#%% 
import numpy as np
import onnx
import os
import glob

from onnx import numpy_helper
#%% Load the ONNX model
model = onnx.load('/home/udhansen/nn_models/image_model/models/clothing_classifier_model.onnx')
#test_data_dir = 'test_data_set_0'
#%% 
# Load inputs
input_file = os.path.join('input0.pb')
tensor = onnx.TensorProto()
with open(input_file, 'rb') as f:
    tensor.ParseFromString(f.read())
x = numpy_helper.to_array(tensor)
# %% Check the generated data
print(x)
print(x.shape)
