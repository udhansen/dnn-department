#%%
"""
This python-script reads the output from the ONNX runtime profiling, and finds the inference time in milliseconds.

Made by: Ulrik Hansen, s195091

"""
#%%
import pandas as pd
import json
import matplotlib.pyplot as plot

# %% CPU - Load files

ONNX_PROFILE_FILE_cpu_run1 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1/onnx_profiling_output_board_cpu_2023-07-27_15-12-58.json'
ONNX_PROFILE_FILE_cpu_run1000 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_2023-07-27_15-15-22.json'

ONNX_PROFILE_FILE_cpu_x1_y1 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x1_y1_2023-08-02_12-06-58.json'
ONNX_PROFILE_FILE_cpu_x1_y2 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x1_y2_2023-08-02_12-07-05.json'
ONNX_PROFILE_FILE_cpu_x1_y4 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x1_y4_2023-08-02_12-07-12.json'

ONNX_PROFILE_FILE_cpu_x2_y1 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x2_y1_2023-08-02_12-07-18.json'
ONNX_PROFILE_FILE_cpu_x2_y2 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x2_y2_2023-08-02_12-07-25.json'
ONNX_PROFILE_FILE_cpu_x2_y4 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x2_y4_2023-08-02_12-07-45.json'

ONNX_PROFILE_FILE_cpu_x4_y1 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x4_y1_2023-08-02_12-07-51.json'
ONNX_PROFILE_FILE_cpu_x4_y2 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x4_y2_2023-08-02_12-07-59.json'
ONNX_PROFILE_FILE_cpu_x4_y4 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_cpu_x4_y4_2023-08-02_12-08-07.json'

# %%
df_cpu_run1 = pd.read_json(ONNX_PROFILE_FILE_cpu_run1).iloc[2:].reset_index(drop=True)
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

# %%
mean_value_cpu_run1 = df_cpu_run1[df_cpu_run1['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_cpu_run1000 = df_cpu_run1000[df_cpu_run1000['name'] =='model_run'].iloc[1:]['dur'].mean()

mean_value_cpu_x1_y1 = df_cpu_x1_y1[df_cpu_x1_y1['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_cpu_x1_y2 = df_cpu_x1_y2[df_cpu_x1_y2['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_cpu_x1_y4 = df_cpu_x1_y4[df_cpu_x1_y4['name'] =='model_run'].iloc[1:]['dur'].mean()

mean_value_cpu_x2_y1 = df_cpu_x2_y1[df_cpu_x2_y1['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_cpu_x2_y2 = df_cpu_x2_y2[df_cpu_x2_y2['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_cpu_x2_y4 = df_cpu_x2_y4[df_cpu_x2_y4['name'] =='model_run'].iloc[1:]['dur'].mean()

mean_value_cpu_x4_y1 = df_cpu_x4_y1[df_cpu_x4_y1['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_cpu_x4_y2 = df_cpu_x4_y2[df_cpu_x4_y2['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_cpu_x4_y4 = df_cpu_x4_y4[df_cpu_x4_y4['name'] =='model_run'].iloc[1:]['dur'].mean()

# %%
df_avgms_data_cpu = pd.DataFrame({
    'Inf. time': [mean_value_cpu_run1, mean_value_cpu_run1000, mean_value_cpu_x1_y1, mean_value_cpu_x1_y2, mean_value_cpu_x1_y4
            , mean_value_cpu_x2_y1, mean_value_cpu_x2_y2, mean_value_cpu_x2_y4, mean_value_cpu_x4_y1
            , mean_value_cpu_x4_y2, mean_value_cpu_x4_y4]},
    index = ['cpu', 'cpu (1000 runs)', 'cpu x:1 y:1', 'cpu x:1 y:2', 'cpu x:1 y:4',
             'cpu x:2 y:1', 'cpu x:2 y:2', 'cpu x:2 y:4'
             , 'cpu x:4 y:1', 'cpu x:4 y:2', 'cpu x:4 y:4']
)

df_avgms_data_cpu = df_avgms_data_cpu / 1000

#%%
print(df_avgms_data_cpu)

# %%
df_avgms_data_cpu.plot(kind="bar")
plot.xlabel('Delegate')
plot.xticks(rotation=45)
plot.ylabel('avg_ms')

# %% NNAPI
ONNX_PROFILE_FILE_nnapi_run1 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1/onnx_profiling_output_board_nnapi_2023-07-27_15-12-37.json'
ONNX_PROFILE_FILE_nnapi_run1000 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_nnapi_2023-07-27_15-15-07.json'

ONNX_PROFILE_FILE_nnapi_x1_y1 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_nnapi_x1_y1_2023-08-02_12-05-31.json'
ONNX_PROFILE_FILE_nnapi_x1_y2 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_nnapi_x1_y2_2023-08-02_12-05-41.json'
ONNX_PROFILE_FILE_nnapi_x1_y4 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_nnapi_x1_y4_2023-08-02_12-05-50.json'

ONNX_PROFILE_FILE_nnapi_x2_y1 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_nnapi_x2_y1_2023-08-02_12-05-59.json'
ONNX_PROFILE_FILE_nnapi_x2_y2 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_nnapi_x1_y2_2023-08-02_12-05-41.json'
ONNX_PROFILE_FILE_nnapi_x2_y4 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_nnapi_x2_y4_2023-08-02_12-06-24.json'

ONNX_PROFILE_FILE_nnapi_x4_y1 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_nnapi_x4_y1_2023-08-02_12-06-31.json'
ONNX_PROFILE_FILE_nnapi_x4_y2 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_nnapi_x4_y2_2023-08-02_12-06-37.json'
ONNX_PROFILE_FILE_nnapi_x4_y4 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_nnapi_x4_y4_2023-08-02_12-06-44.json'

# %%
df_nnapi_run1 = pd.read_json(ONNX_PROFILE_FILE_nnapi_run1).iloc[2:].reset_index(drop=True)
df_nnapi_run1000 = pd.read_json(ONNX_PROFILE_FILE_nnapi_run1000).iloc[2:].reset_index(drop=True)

df_nnapi_x1_y1 = pd.read_json(ONNX_PROFILE_FILE_nnapi_x1_y1).iloc[2:].reset_index(drop=True)
df_nnapi_x1_y2 = pd.read_json(ONNX_PROFILE_FILE_nnapi_x1_y2).iloc[2:].reset_index(drop=True)
df_nnapi_x1_y4 = pd.read_json(ONNX_PROFILE_FILE_nnapi_x1_y4).iloc[2:].reset_index(drop=True)
df_nnapi_x2_y1 = pd.read_json(ONNX_PROFILE_FILE_nnapi_x2_y1).iloc[2:].reset_index(drop=True)
df_nnapi_x2_y2 = pd.read_json(ONNX_PROFILE_FILE_nnapi_x2_y2).iloc[2:].reset_index(drop=True)
df_nnapi_x2_y4 = pd.read_json(ONNX_PROFILE_FILE_nnapi_x2_y4).iloc[2:].reset_index(drop=True)
df_nnapi_x4_y1 = pd.read_json(ONNX_PROFILE_FILE_nnapi_x4_y1).iloc[2:].reset_index(drop=True)
df_nnapi_x4_y2 = pd.read_json(ONNX_PROFILE_FILE_nnapi_x4_y2).iloc[2:].reset_index(drop=True)
df_nnapi_x4_y4 = pd.read_json(ONNX_PROFILE_FILE_nnapi_x4_y4).iloc[2:].reset_index(drop=True)

# %%
mean_value_nnapi_run1 = df_nnapi_run1[df_nnapi_run1['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_nnapi_run1000 = df_nnapi_run1000[df_nnapi_run1000['name'] =='model_run'].iloc[1:]['dur'].mean()

mean_value_nnapi_x1_y1 = df_nnapi_x1_y1[df_nnapi_x1_y1['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_nnapi_x1_y2 = df_nnapi_x1_y2[df_nnapi_x1_y2['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_nnapi_x1_y4 = df_nnapi_x1_y4[df_nnapi_x1_y4['name'] =='model_run'].iloc[1:]['dur'].mean()

mean_value_nnapi_x2_y1 = df_nnapi_x2_y1[df_nnapi_x2_y1['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_nnapi_x2_y2 = df_nnapi_x2_y2[df_nnapi_x2_y2['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_nnapi_x2_y4 = df_nnapi_x2_y4[df_nnapi_x2_y4['name'] =='model_run'].iloc[1:]['dur'].mean()

mean_value_nnapi_x4_y1 = df_nnapi_x4_y1[df_nnapi_x4_y1['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_nnapi_x4_y2 = df_nnapi_x4_y2[df_nnapi_x4_y2['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_nnapi_x4_y4 = df_nnapi_x4_y4[df_nnapi_x4_y4['name'] =='model_run'].iloc[1:]['dur'].mean()

# %%
df_avgms_data_nnapi = pd.DataFrame({
    'Inf. time': [mean_value_nnapi_run1, mean_value_nnapi_run1000, mean_value_nnapi_x1_y1, mean_value_nnapi_x1_y2, mean_value_nnapi_x1_y4
            , mean_value_nnapi_x2_y1, mean_value_nnapi_x2_y2, mean_value_nnapi_x2_y4, mean_value_nnapi_x4_y1
            , mean_value_nnapi_x4_y2, mean_value_nnapi_x4_y4]},
    index = ['nnapi', 'nnapi (1000 runs)', 'nnapi x:1 y:1', 'nnapi x:1 y:2', 'nnapi x:1 y:4',
             'nnapi x:2 y:1', 'nnapi x:2 y:2', 'nnapi x:2 y:4'
             , 'nnapi x:4 y:1', 'nnapi x:4 y:2', 'nnapi x:4 y:4']
)

df_avgms_data_nnapi = df_avgms_data_nnapi / 1000
# %%
print(df_avgms_data_nnapi)

# %%
df_avgms_data_nnapi.plot(kind="bar")
plot.xlabel('Delegate')
plot.xticks(rotation=45)
plot.ylabel('avg_ms')

# %% VSI_NPU
ONNX_PROFILE_FILE_vsi_npu_run1 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1/onnx_profiling_output_board_vsi_npu_2023-07-31_17-46-53.json'
ONNX_PROFILE_FILE_vsi_npu_run1000 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_vsi_npu_2023-07-31_17-48-02.json'

ONNX_PROFILE_FILE_vsi_npu_x1_y1 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_vsi_npu_x1_y1_2023-08-02_12-08-16.json'
ONNX_PROFILE_FILE_vsi_npu_x1_y2 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_vsi_npu_x1_y2_2023-08-02_12-08-24.json'
ONNX_PROFILE_FILE_vsi_npu_x1_y4 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_vsi_npu_x1_y4_2023-08-02_12-08-32.json'

ONNX_PROFILE_FILE_vsi_npu_x2_y1 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_vsi_npu_x2_y1_2023-08-02_12-08-39.json'
ONNX_PROFILE_FILE_vsi_npu_x2_y2 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_vsi_npu_x2_y2_2023-08-02_12-08-46.json'
ONNX_PROFILE_FILE_vsi_npu_x2_y4 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_vsi_npu_x2_y4_2023-08-02_12-08-56.json'

ONNX_PROFILE_FILE_vsi_npu_x4_y1 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_vsi_npu_x4_y1_2023-08-02_12-09-06.json'
ONNX_PROFILE_FILE_vsi_npu_x4_y2 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_vsi_npu_x4_y2_2023-08-02_12-09-13.json'
ONNX_PROFILE_FILE_vsi_npu_x4_y4 = '/home/udhansen/nn_models/image_model/benchmark/board/onnx/profiling_outputs/tflite/run1000/onnx_profiling_output_board_vsi_npu_x4_y4_2023-08-02_12-09-22.json'

# %%
df_vsi_npu_run1 = pd.read_json(ONNX_PROFILE_FILE_vsi_npu_run1).iloc[2:].reset_index(drop=True)
df_vsi_npu_run1000 = pd.read_json(ONNX_PROFILE_FILE_vsi_npu_run1000).iloc[2:].reset_index(drop=True)

df_vsi_npu_x1_y1 = pd.read_json(ONNX_PROFILE_FILE_vsi_npu_x1_y1).iloc[2:].reset_index(drop=True)
df_vsi_npu_x1_y2 = pd.read_json(ONNX_PROFILE_FILE_vsi_npu_x1_y2).iloc[2:].reset_index(drop=True)
df_vsi_npu_x1_y4 = pd.read_json(ONNX_PROFILE_FILE_vsi_npu_x1_y4).iloc[2:].reset_index(drop=True)
df_vsi_npu_x2_y1 = pd.read_json(ONNX_PROFILE_FILE_vsi_npu_x2_y1).iloc[2:].reset_index(drop=True)
df_vsi_npu_x2_y2 = pd.read_json(ONNX_PROFILE_FILE_vsi_npu_x2_y2).iloc[2:].reset_index(drop=True)
df_vsi_npu_x2_y4 = pd.read_json(ONNX_PROFILE_FILE_vsi_npu_x2_y4).iloc[2:].reset_index(drop=True)
df_vsi_npu_x4_y1 = pd.read_json(ONNX_PROFILE_FILE_vsi_npu_x4_y1).iloc[2:].reset_index(drop=True)
df_vsi_npu_x4_y2 = pd.read_json(ONNX_PROFILE_FILE_vsi_npu_x4_y2).iloc[2:].reset_index(drop=True)
df_vsi_npu_x4_y4 = pd.read_json(ONNX_PROFILE_FILE_vsi_npu_x4_y4).iloc[2:].reset_index(drop=True)

# %%
mean_value_vsi_npu_run1 = df_vsi_npu_run1[df_vsi_npu_run1['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_vsi_npu_run1000 = df_vsi_npu_run1000[df_vsi_npu_run1000['name'] =='model_run'].iloc[1:]['dur'].mean()

mean_value_vsi_npu_x1_y1 = df_vsi_npu_x1_y1[df_vsi_npu_x1_y1['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_vsi_npu_x1_y2 = df_vsi_npu_x1_y2[df_vsi_npu_x1_y2['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_vsi_npu_x1_y4 = df_vsi_npu_x1_y4[df_vsi_npu_x1_y4['name'] =='model_run'].iloc[1:]['dur'].mean()

mean_value_vsi_npu_x2_y1 = df_vsi_npu_x2_y1[df_vsi_npu_x2_y1['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_vsi_npu_x2_y2 = df_vsi_npu_x2_y2[df_vsi_npu_x2_y2['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_vsi_npu_x2_y4 = df_vsi_npu_x2_y4[df_vsi_npu_x2_y4['name'] =='model_run'].iloc[1:]['dur'].mean()

mean_value_vsi_npu_x4_y1 = df_vsi_npu_x4_y1[df_vsi_npu_x4_y1['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_vsi_npu_x4_y2 = df_vsi_npu_x4_y2[df_vsi_npu_x4_y2['name'] =='model_run'].iloc[1:]['dur'].mean()
mean_value_vsi_npu_x4_y4 = df_vsi_npu_x4_y4[df_vsi_npu_x4_y4['name'] =='model_run'].iloc[1:]['dur'].mean()

# %%
df_avgms_data_vsi_npu = pd.DataFrame({
    'Inf. time': [mean_value_vsi_npu_run1, mean_value_vsi_npu_run1000, mean_value_vsi_npu_x1_y1, mean_value_vsi_npu_x1_y2, mean_value_vsi_npu_x1_y4
            , mean_value_vsi_npu_x2_y1, mean_value_vsi_npu_x2_y2, mean_value_vsi_npu_x2_y4, mean_value_vsi_npu_x4_y1
            , mean_value_vsi_npu_x4_y2, mean_value_vsi_npu_x4_y4]},
    index = ['vsi_npu', 'vsi_npu (1000 runs)', 'vsi_npu x:1 y:1', 'vsi_npu x:1 y:2', 'vsi_npu x:1 y:4',
             'vsi_npu x:2 y:1', 'vsi_npu x:2 y:2', 'vsi_npu x:2 y:4'
             , 'vsi_npu x:4 y:1', 'vsi_npu x:4 y:2', 'vsi_npu x:4 y:4']
)

# df_avgms_data_vsi_npu = df_avgms_data_vsi_npu / 1000

# %%
df_avgms_data_x_1 = pd.DataFrame({
        'cpu': [mean_value_cpu_x1_y1, mean_value_cpu_x1_y2, mean_value_cpu_x1_y4],
        'nnapi': [mean_value_nnapi_x1_y1, mean_value_nnapi_x1_y2, mean_value_nnapi_x1_y4],
        'vsi_npu': [mean_value_vsi_npu_x1_y1, mean_value_vsi_npu_x1_y2, mean_value_vsi_npu_x1_y4]},
                index = ['y: 1', 'y: 2', 'y: 4']
)

df_avgms_data_x_2 = pd.DataFrame({
        'cpu': [mean_value_cpu_x2_y1, mean_value_cpu_x2_y2, mean_value_cpu_x2_y4],
        'nnapi': [mean_value_nnapi_x2_y1, mean_value_nnapi_x2_y2, mean_value_nnapi_x2_y4],
        'vsi_npu': [mean_value_vsi_npu_x2_y1, mean_value_vsi_npu_x2_y2, mean_value_vsi_npu_x2_y4]},
                index = ['y: 1', 'y: 2', 'y: 4']
)

df_avgms_data_x_4 = pd.DataFrame({
        'cpu': [mean_value_cpu_x4_y1, mean_value_cpu_x4_y2, mean_value_cpu_x4_y4],
        'nnapi': [mean_value_nnapi_x4_y1, mean_value_nnapi_x4_y2, mean_value_nnapi_x4_y4],
        'vsi_npu': [mean_value_vsi_npu_x4_y1, mean_value_vsi_npu_x4_y2, mean_value_vsi_npu_x4_y4]},
                index = ['y: 1', 'y: 2', 'y: 4']
)

# %%
print(mean_value_cpu_run1000)
print(mean_value_nnapi_run1000)
print(mean_value_vsi_npu_run1000)

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
