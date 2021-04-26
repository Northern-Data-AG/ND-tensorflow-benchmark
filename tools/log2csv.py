import os
import re
import glob
import argparse


import pandas as pd


list_test = ['alexnet',
             'inception3', 
             'inception4', 
             'resnet152', 
             'resnet50',
             'vgg16']

# Naming convention
# Key: log name
# Value: ([num_gpus], [names])
# num_gpus: Since each log folder has all the record for different numbers of GPUs, it is convenient to specify the benchmarks you want to pull by listing the num_gpus
# names: rename the experiments so they are easier to undertand

list_system = {
    "7742-A100-SXM4-40GB": ([1, 2, 4, 8], ['NorthernData A100 40GB SXM4', 'NorthernData 2x A100 40GB SXM4', 'NorthernData 4x A100 40GB SXM4', 'NorthernData 8x A100 40GB SXM4'])
    "7402P-Vega_20":  ([1, 2, 4, 8], ['NorthernData — Mi 50', 'NorthernData — 2x Mi 50', 'NorthernData — 4x Mi 50', 'NorthernData — 8x Mi 50'])
  
}


def get_result(path_logs, folder, model):
    folder_path = glob.glob(path_logs + '/' + folder + '/' + model + '*')[0]
    folder_name = folder_path.split('/')[-1]
    batch_size = folder_name.split('-')[-1]
    file_throughput = folder_path + '/throughput/1'
    with open(file_throughput, 'r') as f:
        lines = f.read().splitlines()
        line = lines[-2]
    throughput = line.split(' ')[-1]
    try:
        throughput = int(round(float(throughput)))
    except:
        throughput = 0

    return batch_size, throughput

def create_row_throughput(path_logs, mode, data, precision, key, num_gpu, name, df, is_train=True):
    if is_train:
        if precision == 'fp32':
            folder_fp32 = key + '.logs/' + data + '-' + mode + '-fp32-' + str(num_gpu)+'gpus'
        else:
            folder_fp16 = key + '.logs/' + data + '-' + mode + '-fp16-' + str(num_gpu)+'gpus'
    else:
        if precision == 'fp32':
            folder_fp32 = key + '.logs/' + data + '-' + mode + '-fp32-' + str(num_gpu)+'gpus' + '-inference'
        else:
            folder_fp16 = key + '.logs/' + data + '-' + mode + '-fp16-' + str(num_gpu)+'gpus' + '-inference'

    for model in list_test:
        if precision == 'fp32':
            batch_size, throughput = get_result(path_logs, folder_fp32, model)
        else:
            batch_size, throughput = get_result(path_logs, folder_fp16, model)

        df.at[name, model] = throughput

    df.at[name, 'num_gpu'] = num_gpu


def create_row_batch_size(path_logs, mode, data, precision, key, num_gpu, name, df, is_train=True):
    if is_train:
        if precision == 'fp32':
            folder_fp32 = key + '.logs/' + data + '-' + mode + '-fp32-' + str(num_gpu)+'gpus'
        else:
            folder_fp16 = key + '.logs/' + data + '-' + mode + '-fp16-' + str(num_gpu)+'gpus'
    else:
        if precision == 'fp32':
            folder_fp32 = key + '.logs/' + data + '-' + mode + '-fp32-' + str(num_gpu)+'gpus' + '-inference'
        else:
            folder_fp16 = key + '.logs/' + data + '-' + mode + '-fp16-' + str(num_gpu)+'gpus' + '-inference'
        

    for model in list_test:
        if precision == 'fp32':
            batch_size, throughput = get_result(path_logs, folder_fp32, model)
        else:
            batch_size, throughput = get_result(path_logs, folder_fp16, model)

        df.at[name, model] = int(batch_size) * num_gpu

    df.at[name, 'num_gpu'] = num_gpu



def main():

    parser = argparse.ArgumentParser(description='Gather benchmark results.')

    parser.add_argument('--path', type=str, default='logs',
                        help='path that has the logs')    
    
    parser.add_argument('--mode', type=str, default='replicated',
                        choices=['replicated', 'parameter_server'],
                        help='Method for parameter update')  

    parser.add_argument('--data', type=str, default='syn',
                        choices=['syn', 'real'],
                        help='Choose between synthetic data and real data')

    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16'],
                        help='Choose becnhmark precision')

    args = parser.parse_args()


    columns = []
    columns.append('num_gpu')
    for model in list_test:
        columns.append(model)


    list_row = []
    for key, value in sorted(list_system.items()):  
        for name in value[1]:
            list_row.append(name)

    # Train Throughput
    df_throughput = pd.DataFrame(index=list_row, columns=columns)

    for key in list_system:
        # list_gpus = list_system[key][0]
        for (num_gpu, name) in zip(list_system[key][0], list_system[key][1]):
            create_row_throughput(args.path, args.mode, args.data, args.precision, key, num_gpu, name, df_throughput)

    df_throughput.index.name = 'name_gpu'

    df_throughput.to_csv('tf-train-throughput-' + args.precision + '.csv')

    # # Inference Throughput
    # df_throughput = pd.DataFrame(index=list_row, columns=columns)

    # for key in list_system:
    #     list_gpus = list_system[key]
    #     for num_gpu in list_gpus:
    #         create_row_throughput(args.path, args.mode, key, num_gpu, df_throughput, False)

    # df_throughput.index.name = 'name_gpu'

    # df_throughput.to_csv('tf-inference-throughput-' + precision + '.csv')


    # Train Batch Size
    df_bs = pd.DataFrame(index=list_row, columns=columns)

    for key in list_system:
        for (num_gpu, name) in zip(list_system[key][0], list_system[key][1]):
            create_row_batch_size(args.path, args.mode, args.data, args.precision, key, num_gpu, name, df_bs)

    df_bs.index.name = 'name_gpu'

    df_bs.to_csv('tf-train-bs-' + args.precision + '.csv')


if __name__ == "__main__":
    main()

