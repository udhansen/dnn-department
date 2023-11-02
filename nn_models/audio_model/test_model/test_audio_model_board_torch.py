#%%
# This script tests the audio model with new unseen data. 
# The input data are inside the folder called input_data, where audio recordings in .wav formats I created with AudaCity with own voice are.
# Author: Ulrik, s195091
# This is meant for target board
# Libraries
#%%
import tflite_runtime.interpreter as tflite
import numpy as np
import sys
import wave
import time
import torch

#%%
# Load audio model
TFLITE_FILE_PATH = '/usr/bin/tensorflow-lite-2.6.0/examples/audio_model/keyword_detector_model.tflite'

class_names = ['go', 'no', 'up', 'right', 'left', 'stop', 'down', 'yes']

interpreter = tflite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(input_details)
# print(output_details)

#%%
AUDIO_PATH = '/usr/bin/tensorflow-lite-2.6.0/examples/audio_model/input_data/' + sys.argv[1]

# Load the audio clip 
with wave.open(AUDIO_PATH, 'rb') as audio_file:
    num_channels = audio_file.getnchannels()
    sample_width = audio_file.getsampwidth()
    framerate = audio_file.getframerate()
    num_frames = audio_file.getnframes()
    audio_data = audio_file.readframes(num_frames)

# Convert audio bytes to a numpy array
waveform = np.frombuffer(audio_data, dtype=np.int16)

# Ensure waveform length is 16000
desired_length = 16000
if len(waveform) < desired_length:
    padding = np.zeros(desired_length - len(waveform), dtype=np.float32)
    waveform = np.concatenate((waveform, padding))
elif len(waveform) > desired_length:
    waveform = waveform[:desired_length]

# Normalize the waveform to the range [-1, 1]
normalized_waveform = waveform / np.iinfo(np.int16).max

# Convert the normalized waveform to a PyTorch tensor and add a batch dimension
audio_data = torch.tensor(normalized_waveform, dtype=torch.float32)
audio_data = audio_data.unsqueeze(0)

# print(audio_data.shape)
# print(sampling_rate)

# %%
# Get inference time / latency
start_time = time.time()

# Feed the model with the audio file and get a prediction
interpreter.set_tensor(input_details[0]['index'], audio_data)
interpreter.invoke()
output_data0 = interpreter.get_tensor(output_details[0]['index'])
output_data1 = interpreter.get_tensor(output_details[1]['index'])
output_data2 = interpreter.get_tensor(output_details[2]['index'])
print("\nOutput array: ", output_data0)
print("\nOutput array: ", output_data1)
print("\nOutput array: ", output_data2)

# Get inference time
end_time = time.time()
get_inference_time = end_time - start_time

print("Inference time: {:.4f} seconds".format(get_inference_time))
# %%
