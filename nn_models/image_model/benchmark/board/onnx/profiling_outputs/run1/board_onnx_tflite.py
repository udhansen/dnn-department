#%%
"""
This script creates a bar plot of the results from ONNX profiling for the audio model. It takes the mean of all sessions named 'model_run'.

Author: Ulrik, s195091 

"""
#%%
import pandas as pd
import json
import matplotlib.pyplot as plot

# %% CPU - Load profiling files (.json format)
ONNX_PROFILE_FILE_cpu_run1000 = '/home/udhansen/nn_models/audio_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_2023-08-02_18-24-56.json'

ONNX_PROFILE_FILE_cpu_x1_y1 = '/home/udhansen/nn_models/audio_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x1_y1_2023-08-02_18-25-11.json'
ONNX_PROFILE_FILE_cpu_x1_y2 = '/home/udhansen/nn_models/audio_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x1_y2_2023-08-02_18-25-28.json'
ONNX_PROFILE_FILE_cpu_x1_y4 = '/home/udhansen/nn_models/audio_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x1_y4_2023-08-02_18-25-41.json'

ONNX_PROFILE_FILE_cpu_x2_y1 = '/home/udhansen/nn_models/audio_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x2_y1_2023-08-02_18-25-56.json'
ONNX_PROFILE_FILE_cpu_x2_y2 = '/home/udhansen/nn_models/audio_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x2_y2_2023-08-02_18-26-12.json'
ONNX_PROFILE_FILE_cpu_x2_y4 = '/home/udhansen/nn_models/audio_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x2_y4_2023-08-02_18-26-32.json'

ONNX_PROFILE_FILE_cpu_x4_y1 = '/home/udhansen/nn_models/audio_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x4_y1_2023-08-02_18-26-46.json'
ONNX_PROFILE_FILE_cpu_x4_y2 = '/home/udhansen/nn_models/audio_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x4_y2_2023-08-02_18-27-03.json'
ONNX_PROFILE_FILE_cpu_x4_y4 = '/home/udhansen/nn_models/audio_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x4_y4_2023-08-02_18-27-17.json'

# %% Read the .json's files and skip init phase values
df_cpu_run1000 = pd.read_json(ONNX_PROFILE_FILE_cpu_run1000).iloc[2:].reset_index(drop=True)

df_cpu_x1_y1 = pd.read_json(ONNX_PROFILE_FILE_cpu_x1_y1).iloc[2:].reset_index(drop=True)
df_cpu_x1_y2 = pd.read_json(ONNX_PROFILE_FILE_cpu_x1_y2).iloc[2:].reset_index(drop=True)
df_cpu_x1_y4 = pd.read_json(ONNX_PROFILE_FILE_cpu_x1_y4).iloc[2:].reset_index(drop=True)
df_cpu_x2_y1 = pd.read_json(ONNX_PROFILE_FILE_cpu_x2_y1).iloc[2:].reset_index(drop=True)
df_cpu_x2_y2 = pd.read_json(ONNX_PROFILE_FILE_cpu_x2_y2).iloc[2:].reset_index(drop=True)
df_cpu_x2_y4 = pd.read_json(ONNX_PROFILE_FILE_cpu_x2_y4).iloc[2:].reset_index(drop=True)
df_cpu_x4_y1 = pd.read_json(ONNX_PROFILE_FILE_cpu_x4_y1).iloc[2:].reset_index(drop=True)
df_cpu_x4_y2 = pd.read_json(ONNX_PROFILE_FILE_cpu_x4_y2).iloc[2:].reset_index(drop=True)
df_cpu_x4_y4 = pd.read_json(ONNX_PROFILE_FILE_cpu_x4_y4).iloc[2:].reset_index(drop=True)

# %% Determine the mean value and convert it from microseconds into milliseconds
mean_value_cpu_run1000 = df_cpu_run1000[df_cpu_run1000['name'] =='model_run'].iloc[1:]['dur'].mean()
#mean_value_cpu_run1000 = mean_value_cpu_run1000 / 1000

mean_value_cpu_x1_y1 = df_cpu_x1_y1[df_cpu_x1_y1['name'] =='model_run'].iloc[1:]['dur'].mean()
#mean_value_cpu_x1_y1 = mean_value_cpu_x1_y1 / 1000

mean_value_cpu_x1_y2 = df_cpu_x1_y2[df_cpu_x1_y2['name'] =='model_run'].iloc[1:]['dur'].mean()
#mean_value_cpu_x1_y2 = mean_value_cpu_x1_y2 / 1000

mean_value_cpu_x1_y4 = df_cpu_x1_y4[df_cpu_x1_y4['name'] =='model_run'].iloc[1:]['dur'].mean()
#mean_value_cpu_x1_y4 = mean_value_cpu_x1_y4 / 1000

mean_value_cpu_x2_y1 = df_cpu_x2_y1[df_cpu_x2_y1['name'] =='model_run'].iloc[1:]['dur'].mean()
#mean_value_cpu_x2_y1 = mean_value_cpu_x1_y1 / 1000

mean_value_cpu_x2_y2 = df_cpu_x2_y2[df_cpu_x2_y2['name'] =='model_run'].iloc[1:]['dur'].mean()
#mean_value_cpu_x2_y2 = mean_value_cpu_x1_y2 / 1000

mean_value_cpu_x2_y4 = df_cpu_x2_y4[df_cpu_x2_y4['name'] =='model_run'].iloc[1:]['dur'].mean()
#mean_value_cpu_x2_y4 = mean_value_cpu_x1_y4 / 1000

mean_value_cpu_x4_y1 = df_cpu_x4_y1[df_cpu_x4_y1['name'] =='model_run'].iloc[1:]['dur'].mean()
#mean_value_cpu_x4_y1 = mean_value_cpu_x1_y1 / 1000

mean_value_cpu_x4_y2 = df_cpu_x4_y2[df_cpu_x4_y2['name'] =='model_run'].iloc[1:]['dur'].mean()
#mean_value_cpu_x4_y2 = mean_value_cpu_x1_y2 / 1000

mean_value_cpu_x4_y4 = df_cpu_x4_y4[df_cpu_x4_y4['name'] =='model_run'].iloc[1:]['dur'].mean()
#mean_value_cpu_x4_y4 = mean_value_cpu_x1_y4 / 1000

# %% Create a dataframe with all mean values in order to plot it
df_avgms_data_cpu = pd.DataFrame({
    'Inf. time': [mean_value_cpu_run1000, mean_value_cpu_x1_y1, mean_value_cpu_x1_y2, mean_value_cpu_x1_y4
            , mean_value_cpu_x2_y1, mean_value_cpu_x2_y2, mean_value_cpu_x2_y4, mean_value_cpu_x4_y1
            , mean_value_cpu_x4_y2, mean_value_cpu_x4_y4]},
    index = ['cpu (1000 runs)', 'cpu x:1 y:1', 'cpu x:1 y:2', 'cpu x:1 y:4',
             'cpu x:2 y:1', 'cpu x:2 y:2', 'cpu x:2 y:4'
             , 'cpu x:4 y:1', 'cpu x:4 y:2', 'cpu x:4 y:4']
)

#%% See the values
print(df_avgms_data_cpu)

# %% Generate the plot
df_avgms_data_cpu.plot(kind="bar")
plot.xlabel('Delegate')
plot.xticks(rotation=45)
plot.ylabel('avg_ms')

# %%
df_avgms_data_x_1 = pd.DataFrame({
        'cpu': [mean_value_cpu_x1_y1, mean_value_cpu_x1_y2, mean_value_cpu_x1_y4]},
                index = ['y: 1', 'y: 2', 'y: 4']
)

df_avgms_data_x_2 = pd.DataFrame({
        'cpu': [mean_value_cpu_x2_y1, mean_value_cpu_x2_y2, mean_value_cpu_x2_y4]},
                index = ['y: 1', 'y: 2', 'y: 4']
)

df_avgms_data_x_4 = pd.DataFrame({
        'cpu': [mean_value_cpu_x4_y1, mean_value_cpu_x4_y2, mean_value_cpu_x4_y4]},
                index = ['y: 1', 'y: 2', 'y: 4']

)
# %%
print(mean_value_cpu_run1000)

print(df_avgms_data_x_1)
print(df_avgms_data_x_2)
print(df_avgms_data_x_4)

# %%
df_avgms_data_x_1.plot(kind="bar")
plot.title('x: 1')
plot.xticks(rotation=45)
plot.ylabel('Time (us)')
# %%
df_avgms_data_x_2.plot(kind="bar")
plot.title('x: 2')
plot.xticks(rotation=45)
plot.ylabel('Time (us)')
# %%
df_avgms_data_x_4.plot(kind="bar")
plot.title('x: 4')
plot.xticks(rotation=45)
plot.ylabel('Time (us)')
# %%