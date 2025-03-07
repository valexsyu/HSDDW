
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd

GG=10
GG1=10
bins = 10  # 根據範圍0~i設置bins的數量
def update_file_path(experiment, dataset) :

    ROOT="/work/valex1377/LLMSpeculativeSampling/"
    LENGTH_FILE_NAME={
        '70b':  ROOT+f'experiments/{dataset}/llama-2-70b-chat_topk0p0_fp16_2048tok/length_output_tokens.csv',
        # '70b_7b' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_1b' : '/work/valex1377/LLMSpeculativeSampling/new_experiments/mt_bench/10_70b_1b_topkp0_fp16_2048tok/sp_output_length_10_70b_1b_topkp0_fp16_2048tok.csv',
        '70b_68m' : '/work/valex1377/LLMSpeculativeSampling/experiments/mt_bench/sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_68m': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_1b_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok.csv',
    }
    TIME_FILE_NAME={
        '70b':  ROOT+f'experiments/{dataset}/llama-2-70b-chat_topk0p0_fp16_2048tok/time.csv',
        # '70b_7b': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_1b' : '/work/valex1377/LLMSpeculativeSampling/new_experiments/mt_bench/10_70b_1b_topkp0_fp16_2048tok/sp_time_10_70b_1b_topkp0_fp16_2048tok.csv',
        '70b_68m' : '/work/valex1377/LLMSpeculativeSampling/experiments/mt_bench/sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_time_sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_68m': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_1b_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok.csv',
    }
    
    ACCEPTED_FILE_NAME={
        '70b':  ROOT+f'experiments/{dataset}/llama-2-70b-chat_topk0p0_fp16_2048tok/time.csv',
        # '70b_7b': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_accepted_sequence_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_accepted_sequence_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_1b' : '/work/valex1377/LLMSpeculativeSampling/new_experiments/mt_bench/10_70b_1b_topkp0_fp16_2048tok/sp_accepted_sequence_10_70b_1b_topkp0_fp16_2048tok.csv',
        '70b_68m' :'/work/valex1377/LLMSpeculativeSampling/experiments/mt_bench/sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_accepted_sequence_sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_accepted_sequence_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_68m': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_accepted_sequence_sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_accepted_sequence_sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_accepted_sequence_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_1b_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_accepted_sequence_sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok.csv',
    }    
    return TIME_FILE_NAME[experiment], LENGTH_FILE_NAME[experiment] , ACCEPTED_FILE_NAME[experiment]




# 讀取CSV檔案並將數據存儲在列表中
def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            data.extend(map(int, row))
    return data

# 計算每個bin的總數量
# 計算每個bin的總數量
def calculate_histogram(data, bins):
    histogram = [0] * (bins + 1)
    for value in data:
        if value > bins:
            value = bins
        histogram[value] += 1
    return histogram


DATASETS = ["mt_bench", ]
EXPERIMENTS_BL = ['70b']
EXPERIMENTS = ["70b_7b", "70b_1b", "70b_68m"]
SPECDEC={
    'mt_bench': None , 
    'humaneval':20.61/18.55 ,
    'gsm8k':20.95/19.14 , 
    'alpaca': 18.88/17.62 ,
}
PERAL={
    'mt_bench': 2.48 , 
    'humaneval':3.01 ,
    'gsm8k': 2.87 , 
    'alpaca': None ,    
}

# 創建一個 DataFrame 來存儲 baseline 的結果
baseline_df = pd.DataFrame(columns=DATASETS, index=EXPERIMENTS_BL)
# 創建一個 DataFrame 來存儲其他 experiment 的結果
experiment_df = pd.DataFrame(columns=DATASETS, index=EXPERIMENTS)


# 計算 baseline 的結果
for experiment in EXPERIMENTS_BL:
    for dataset in DATASETS:
        sequence_time_path, sequence_length_path , accetped_seq_path = update_file_path(experiment, dataset)
        avg_value = 0
        if os.path.exists(sequence_time_path) and os.path.exists(sequence_length_path):
            with open(sequence_time_path, 'r') as f:
                time_values = [float(line.strip()) for line in f.readlines() if line.strip()]
            with open(sequence_length_path, 'r') as f:
                length_values = [float(line.strip()) for line in f.readlines() if line.strip()]
            
            if time_values and length_values:
                avg_value = np.sum(np.array(length_values)) / np.sum(np.array(time_values))
            baseline_df.at[experiment, dataset] = avg_value  # 存入 DataFrame
        else:
            baseline_df.at[experiment, dataset] = "N/A"


# 主程式
data=[]
hist=[]
ratios=[]
# 計算其他 experiment 的結果並與 baseline 做對比
for experiment in EXPERIMENTS:
    
    for dataset in DATASETS:
        sequence_time_path, sequence_length_path, accetped_seq_path = update_file_path(experiment, dataset)
        avg_value = 0
        if os.path.exists(sequence_time_path) and os.path.exists(sequence_length_path):
            with open(sequence_time_path, 'r') as f:
                time_values = [float(line.strip()) for line in f.readlines() if line.strip()]
            with open(sequence_length_path, 'r') as f:
                length_values = [float(line.strip()) for line in f.readlines() if line.strip()]

            if time_values and length_values:
                
                tot_time_array = np.array(length_values)/np.array(time_values) 
                avg_value = np.sum(np.array(length_values))/np.sum(tot_time_array)  # 計算平均值                
                # 計算與 baseline 的比例
                bl_value = baseline_df.at[EXPERIMENTS_BL[0], dataset]
                if bl_value != "N/A" and bl_value != 0:
                    ratio = avg_value / float(bl_value)
                else:
                    ratio = "N/A"
                experiment_df.at[experiment, dataset] = f"{round(avg_value, 2)} ({round(ratio,2)}x)"
        else:
            experiment_df.at[experiment, dataset] = "N/A"

    data_output = read_csv(accetped_seq_path)
    data.append(data_output)
    hist.append(calculate_histogram(data_output, bins))
    ratios.append(ratio)



# 計算每個bin的總數量
def calculate_histogram(data, bins):
    histogram = [0] * (bins + 1)
    for value in data:
        if value > bins:
            value = bins
        histogram[value] += 1
    return histogram

# 繪製直方圖
def plot_histogram(hist, bins,experiments,ratios):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
    num_experiment = len(experiments)
    width = 0.9/num_experiment
    shift = 1
    aligns=['center','center','center','center']
    for j, experiment in enumerate(experiments):    
        x = np.arange(bins + 1)
        ax1.bar(x+(j-shift)*width, hist[j], width, label=experiment+f"({round(ratios[j],2)}x)", align=aligns[j])
    ax1.set_xlabel('Accepted Number Per Iteration',fontsize=14)
    ax1.set_ylabel('Total Count',fontsize=14)
    ax1.set_title('Speculative Decoding with Different Model Sizes',fontsize=16)
    ax1.legend(fontsize=14)


    # 儲存圖像
    output_file = "/work/valex1377/LLMSpeculativeSampling/scripts/hir_sd_w_dy_wdw/figs/1_motivation.png"
    plt.savefig(output_file)
    print(f"Finish save plot : {output_file}")



plot_histogram(hist, bins,EXPERIMENTS,ratios)