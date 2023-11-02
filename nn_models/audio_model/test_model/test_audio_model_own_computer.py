#%%
# This script tests the audio model with new unseen data. 
# The input data are inside the folder called input_data, where audio recordings in .wav formats I created with AudaCity with own voice are.
# Author: Ulrik, s195091

# Libraries
#%%
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import time

# Audio processing library
import librosa

#%%
# Load audio model
TFLITE_FILE_PATH = '/home/udhansen/nn_models/audio_model/models/tflite/keyword_detector_model.tflite'

class_names = ['go', 'no', 'up', 'right', 'left', 'stop', 'down', 'yes']

interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

#%%
AUDIO_PATH = '/home/udhansen/nn_models/audio_model/test_model/input_data/' + sys.argv[1]
#AUDIO_PATH = '/home/udhansen/nn_models/audio_model/test_model/input_data/yes.wav'

# Load audio clip
audio_data, sampling_rate = librosa.load(AUDIO_PATH, sr=16000)
audio_data = librosa.util.fix_length(audio_data, size = 16000)
audio_data = audio_data[np.newaxis, ...]

print(audio_data.shape)
print(sampling_rate)

#%%
# Plot audio recording in time domain
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio_data, sr=sampling_rate)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.tight_layout()
plt.show()

#%%
# Plot the corresponding spectrogram
spectrogram = librosa.stft(audio_data[0])
spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
plt.figure(figsize=(10, 6))
librosa.display.specshow(spectrogram_db, sr=sampling_rate, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()
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
# %%
# Get inference time
end_time = time.time()
inference_time = end_time - start_time

print("Inference time: {:.4f} seconds\n".format(inference_time))