# This script is used to process the data from the TFLite profiling outputs in order to make more readable for pandas library 
#
# This is used for data taken from the host pc
#
# Author: Ulrik, s19591

# Libraries
import pandas as pd
import sys

# Get user information
user_input_benchmark_truns = input("Enter the number of runs (run1/run1000)?: ")
TFLITE_PROFILING_FILE_PATH = '/home/udhansen/nn_models/image_model/benchmark/own_computer/tflite/profiling_outputs/' + user_input_benchmark_truns + '/' + sys.argv[1]

# Open file - read line by line - Title reading
with open(TFLITE_PROFILING_FILE_PATH, 'r') as f:
    title = []
    for line in f:
        if line.startswith('='):
            title.append(line.replace('=','').strip().replace(' ','_'))

# Open file to look for data in between titles/headers
with open(TFLITE_PROFILING_FILE_PATH, 'r') as file:
    lines = file.readlines()
    values1 = []
    values2 = []
    values3 = []
    values4 = []
    values5 = []
    values6 = []

    for line_number in [2, 3, 4]:
        if line_number <= len(lines):
            values1.append(lines[line_number].strip().split(','))
            #print(values1)

    for line_number in [7, 8, 9]:
        if line_number <= len(lines):
            values2.append(lines[line_number].strip().split(','))
            #print(values2)
    
    for line_number in [13, 14, 15]:
        if line_number <= len(lines):
            values3.append(lines[line_number].strip().split(','))
            #print(values3)

    values4.append(lines[24].strip().split(','))
    for line_number in [25, 26, 27]:
        if line_number <= len(lines):
            values4.append(lines[line_number].strip().replace(',','',1).split(','))
            #print(values4)

    values5.append(lines[30].strip().split(','))
    for line_number in [31, 32, 33]:
        if line_number <= len(lines):
            values5.append(lines[line_number].strip().replace(',','',1).split(','))
            #print(values5)

    values6.append(lines[37].strip().split(','))
    for line_number in [38, 39]:
        if line_number <= len(lines):
            values6.append(lines[line_number].strip().replace(',','',1).split(','))
            #print(values6)

# Benchmark data - init
benchmark_data_init = {
    title[0]: values1,
    title[1]: values2,
    title[2]: values3,
}

# # Benchmark data - reg
benchmark_data_reg = {
    title[3]: values4,
    title[4]: values5,
    title[5]: values6
}

# Ask for the location to read and store the benchmark
user_input_benchmark_dtype = input("Enter the name of the delegate: ")
user_input_benchmark_numtype = input("Enter the amount of threads used: ")

# Create .csv files for each header for init phase
for key_init, value_init in benchmark_data_init.items():
    
    filename = f"{key_init}.csv"

    with open('/home/udhansen/nn_models/image_model/benchmark/own_computer/tflite/benchmark_data_init/' + user_input_benchmark_truns + '/' + user_input_benchmark_dtype + '/' + user_input_benchmark_numtype + '/' + filename, 'w') as file:

        header = str(key_init)
        file.write(header + "\n")

        for val in value_init:
            row_str = str(val).strip("[]").replace("'","")
            file.write(row_str + "\n")

for key_reg, value_reg in benchmark_data_reg.items():

    filename = f"{key_reg}.csv"

    with open('/home/udhansen/nn_models/image_model/benchmark/own_computer/tflite/benchmark_data_reg/' + user_input_benchmark_truns + '/' + user_input_benchmark_dtype + '/' + user_input_benchmark_numtype + '/' + filename, 'w') as file:

        header = str(key_reg)
        file.write(header + "\n")

        for val in value_reg:
            row_str = str(val).strip("[]").replace("'","")
            file.write(row_str + "\n")

# Pandas
print('\nThis is the init phase:')
df_init_RO = pd.read_csv('/home/udhansen/nn_models/image_model/benchmark/own_computer/tflite/benchmark_data_init/' + user_input_benchmark_truns + '/' + user_input_benchmark_dtype + '/' + user_input_benchmark_numtype + '/' + 'Run_Order.csv')
df_init_CT = pd.read_csv('/home/udhansen/nn_models/image_model/benchmark/own_computer/tflite/benchmark_data_init/' + user_input_benchmark_truns + '/' + user_input_benchmark_dtype + '/' + user_input_benchmark_numtype + '/' + 'Top_by_Computation_Time.csv')
df_init_S = pd.read_csv('/home/udhansen/nn_models/image_model/benchmark/own_computer/tflite/benchmark_data_init/' + user_input_benchmark_truns + '/' + user_input_benchmark_dtype + '/' + user_input_benchmark_numtype + '/' + 'Summary_by_node_type.csv')

print(df_init_RO, '\n')
print(df_init_CT, '\n')
print(df_init_S, '\n')


print('\nThis is the regular phase:')
df_reg_RO = pd.read_csv('/home/udhansen/nn_models/image_model/benchmark/own_computer/tflite/benchmark_data_reg/' + user_input_benchmark_truns + '/' + user_input_benchmark_dtype + '/' + user_input_benchmark_numtype + '/' + 'Run_Order.csv')
df_reg_CT = pd.read_csv('/home/udhansen/nn_models/image_model/benchmark/own_computer/tflite/benchmark_data_reg/' + user_input_benchmark_truns + '/' + user_input_benchmark_dtype + '/' + user_input_benchmark_numtype + '/' + 'Top_by_Computation_Time.csv')
df_reg_S = pd.read_csv('/home/udhansen/nn_models/image_model/benchmark/own_computer/tflite/benchmark_data_reg/' + user_input_benchmark_truns + '/' + user_input_benchmark_dtype + '/' + user_input_benchmark_numtype + '/' + 'Summary_by_node_type.csv')

print(df_reg_RO, '\n')
print(df_reg_CT, '\n')
print(df_reg_S, '\n')
