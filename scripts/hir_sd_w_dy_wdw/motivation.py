

####======================================dynamic_gamma_experiment_comparison_plot================
## Time vs gamma among different experiments



import os
import numpy as np
import matplotlib.pyplot as plt


# 定義範圍
dataset_name = "mt_bench"
# dataset_name = "humaneval"
experiments_baseline = ['70b']
experiments = ["70b_7b", "70b_1b", "70b_68m"]

# 初始化一個字典來儲存每個 experiment 的平均值
speed_averages = {}
dynamic_time_averages = {}  # 儲存 dynamic gamma 的結果
# 定義資料夾路徑
bins = 10  # 根據範圍0~i設置bins的數量
EXPERIIMENTS_FOLDER_PATH={
    '70b':  '/work/valex1377/LLMSpeculativeSampling/experiments/mt_bench/70b_fp16_2048tok_record_only' ,
    '70b_7b': '/work/valex1377/LLMSpeculativeSampling/new_experiments/mt_bench/10_70b_7b_topkp0_fp16_2048tok' ,
    '70b_1b': '/work/valex1377/LLMSpeculativeSampling/new_experiments/mt_bench/10_70b_1b_topkp0_fp16_2048tok' ,
    '70b_68m': '/work/valex1377/LLMSpeculativeSampling/new_experiments/mt_bench/10_70b_68m_topkp0_fp16_2048tok'
}


import csv
import matplotlib.pyplot as plt




FILE_NAME_ACCEPT_BASE =(
    f"""entropy_mesure_accepted_sequence_{{experiment}}_fp16_2048tok_record_only.csv"""
)
FILE_NAME_ACCEPT = (
    f"""sp_accepted_sequence_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
)

FILE_NAME_SEQ_LEN_BASE =(
    f"""entropy_mesure_output_length_{{experiment}}_fp16_2048tok_record_only.csv"""
)
FILE_NAME_SEQ_LEN = (
    f"""sp_output_length_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
)

FILE_NAME_TIME_BASE =(
    f"""entropy_mesure_time_{{experiment}}_fp16_2048tok_record_only.csv"""
)
FILE_NAME_TIME = (
    f"""sp_time_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
)

ACCEPTED_FILE_NAME={
    '70b':  FILE_NAME_ACCEPT_BASE,
    '70b_7b': FILE_NAME_ACCEPT ,
    '70b_1b': FILE_NAME_ACCEPT ,
    '70b_68m': FILE_NAME_ACCEPT
}

LENGTH_FILE_NAME={
    '70b':  FILE_NAME_SEQ_LEN_BASE,
    '70b_7b': FILE_NAME_SEQ_LEN ,
    '70b_1b': FILE_NAME_SEQ_LEN ,
    '70b_68m': FILE_NAME_SEQ_LEN
}
TIME_FILE_NAME={
    '70b':  FILE_NAME_TIME_BASE,
    '70b_7b': FILE_NAME_TIME ,
    '70b_1b': FILE_NAME_TIME ,
    '70b_68m': FILE_NAME_TIME
}


# 讀取CSV檔案並將數據存儲在列表中
def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            data.extend(map(int, row))
    return data

# 計算每個bin的總數量
def calculate_histogram(data, bins):
    histogram = [0] * (bins + 1)
    for value in data:
        if value > bins:
            value = bins
        histogram[value] += 1
    return histogram

# 繪製直方圖
def plot_histogram(hist, bins,experiments,speed_averages, baseline_time_averages):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
    num_experiment = len(experiments)
    width = 0.9/num_experiment
    shift = 1
    models = list(speed_averages.keys())
    values = [speed_averages[model] for model in models]
    baseline_model = list(baseline_time_averages.keys())[0]
    baseline_value = baseline_time_averages[baseline_model]    

    aligns=['center','center','center','center']
    for j, experiment in enumerate(experiments):    
        x = np.arange(bins + 1)
        speed=round(values[j]/baseline_value, 3)
        ax1.bar(x+(j-shift)*width, hist[j], width, label=experiment+f"({speed}x)", align=aligns[j])
    ax1.set_xlabel('Accepted Number Per Iteration',fontsize=14)
    ax1.set_ylabel('Totla Count',fontsize=14)
    ax1.set_title('Speculative Decoding with Different Model Sizes',fontsize=16)
    ax1.legend(fontsize=14)


    # 儲存圖像
    output_file = "/work/valex1377/LLMSpeculativeSampling/scripts/hir_sd_w_dy_wdw/figs/1_motivation.png"
    plt.savefig(output_file)
    print(f"Finish save plot : {output_file}")




# 主程式
data=[]
hist=[]
for j, experiment in enumerate(experiments):

    file_path = os.path.join(
        EXPERIIMENTS_FOLDER_PATH[experiment], 
        ACCEPTED_FILE_NAME[experiment].format(experiment=experiment),
    )    

    data.append(read_csv(file_path))
    hist.append(calculate_histogram(data[j], bins))


for experiment in experiments:
    averages = {}  # 儲存當前 experiment 的平均值
    dynamic_averages = {}  # 儲存當前 experiment 在 dynamic gamma 下的平均值

    sequence_time_path = os.path.join(
        EXPERIIMENTS_FOLDER_PATH[experiment], 
        TIME_FILE_NAME[experiment].format(experiment=experiment),
    )    
    sequence_length_path = os.path.join(
        EXPERIIMENTS_FOLDER_PATH[experiment], 
        LENGTH_FILE_NAME[experiment].format(experiment=experiment),
    )       
        # 檢查標準實驗檔案是否存在並計算平均值

    # if experiment not in ['70b_7b_68m','70b_7b_68m_adv'] :
    if os.path.exists(sequence_time_path) and os.path.exists(sequence_length_path):
        with open(sequence_time_path, 'r') as f:
            data = f.readlines()
        time_values = [float(line.strip()) for line in data if line.strip()]  # 將資料轉換為浮點數列表
        with open(sequence_length_path, 'r') as f:
            data = f.readlines()
        length_values = [float(line.strip()) for line in data if line.strip()]  # 將資料轉換為浮點數列表
        
        if time_values or length_values:
            tot_time_array = np.array(length_values)/np.array(time_values) 
            avg_value = np.sum(np.array(length_values))/np.sum(tot_time_array)  # 計算平均值
    else:
        if os.path.exists(sequence_time_path) is False:
            print(f"No valid data in file: {sequence_time_path.split(f'/{dataset_name}/')[:]}")
        elif os.path.exists(sequence_length_path) is False:
            print(f"No valid data in file: {sequence_length_path.split(f'/{dataset_name}/')[:]}")
        

    # 儲存實驗結果
    speed_averages[experiment] = avg_value




baseline_time_averages = {}  # 儲存當前 experiment 的平均值
for idx, experiment in enumerate(experiments_baseline):


    sequence_time_path = os.path.join(
        EXPERIIMENTS_FOLDER_PATH[experiment], 
        TIME_FILE_NAME[experiment].format(experiment=experiment),
    )    
    sequence_length_path = os.path.join(
        EXPERIIMENTS_FOLDER_PATH[experiment], 
        LENGTH_FILE_NAME[experiment].format(experiment=experiment),
    )     
    
    # 檢查標準實驗檔案是否存在並計算平均值
    if os.path.exists(sequence_time_path) and os.path.exists(sequence_length_path):
        with open(sequence_time_path, 'r') as f:
            data = f.readlines()
        time_values = [float(line.strip()) for line in data if line.strip()]  # 將資料轉換為浮點數列表
        with open(sequence_length_path, 'r') as f:
            data = f.readlines()
        length_values = [float(line.strip()) for line in data if line.strip()]  # 將資料轉換為浮點數列表
        
        if time_values or length_values:
            tot_time_array = np.array(length_values)/np.array(time_values) 
            avg_value = np.sum(np.array(length_values))/np.sum(tot_time_array)  # 計算平均值
        baseline_time_averages[experiment] = avg_value        
            
    else:
        if os.path.exists(sequence_time_path) is False:
            print(f"No valid data in file: {sequence_time_path.split(f'/{dataset_name}/')[-1]}")
        elif os.path.exists(sequence_length_path) is False:
            print(f"No valid data in file: {sequence_length_path.split(f'/{dataset_name}/')[-1]}")           
        baseline_time_averages[experiment] = None   
        # 儲存實驗結果

# print(f"speed_averages:{speed_averages}")
# print(f"baseline_time_averages:{baseline_time_averages}")


plot_histogram(hist, bins,experiments,speed_averages,baseline_time_averages)


