
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd

GG=10
GG1=10
bins = 10  # 根據範圍0~i設置bins的數量
def update_file_path(experiment, dataset, gamma) :

    ROOT="/work/valex1377/LLMSpeculativeSampling/"
    LENGTH_FILE_NAME={
        '70b':  ROOT+f'experiments/{dataset}/llama-2-70b-chat_topk0p0_fp16_2048tok/length_output_tokens.csv',
        # '70b_7b' : ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_output_length_sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b' : ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_output_length_sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_1b' : '/work/valex1377/LLMSpeculativeSampling/new_experiments/mt_bench/{gamma}_70b_1b_topkp0_fp16_2048tok/sp_output_length_{gamma}_70b_1b_topkp0_fp16_2048tok.csv',
        '70b_68m' : '/work/valex1377/LLMSpeculativeSampling/experiments/mt_bench/sp_dy_gamma_{gamma}_70b_68m_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_output_length_sp_dy_gamma_{gamma}_70b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_output_length_sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_68m': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_{gamma}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_{gamma}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_{gamma}_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_1b_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_1b_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_{gamma}_70b_7b_1b_adv_topkp0_fp16_2048tok.csv',
    }
    TIME_FILE_NAME={
        '70b':  ROOT+f'experiments/{dataset}/llama-2-70b-chat_topk0p0_fp16_2048tok/time.csv',
        # '70b_7b': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_time_sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_time_sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_1b' : '/work/valex1377/LLMSpeculativeSampling/new_experiments/mt_bench/{gamma}_70b_1b_topkp0_fp16_2048tok/sp_time_{gamma}_70b_1b_topkp0_fp16_2048tok.csv',
        '70b_68m' : '/work/valex1377/LLMSpeculativeSampling/experiments/mt_bench/sp_dy_gamma_{gamma}_70b_68m_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_time_sp_dy_gamma_{gamma}_70b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_time_sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_68m': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_{gamma}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_{gamma}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_{gamma}_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_1b_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_1b_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_{gamma}_70b_7b_1b_adv_topkp0_fp16_2048tok.csv',
    }
    
    ACCEPTED_FILE_NAME={
        '70b':  ROOT+f'experiments/{dataset}/llama-2-70b-chat_topk0p0_fp16_2048tok/time.csv',
        # '70b_7b': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_accepted_sequence_sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_accepted_sequence_sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_1b' : '/work/valex1377/LLMSpeculativeSampling/new_experiments/mt_bench/{gamma}_70b_1b_topkp0_fp16_2048tok/sp_accepted_sequence_{gamma}_70b_1b_topkp0_fp16_2048tok.csv',
        '70b_68m' :'/work/valex1377/LLMSpeculativeSampling/experiments/mt_bench/sp_dy_gamma_{gamma}_70b_68m_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_accepted_sequence_sp_dy_gamma_{gamma}_70b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_accepted_sequence_sp_dy_gamma_{gamma}_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_68m': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_accepted_sequence_sp_dy_gamma_{gamma}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_accepted_sequence_sp_dy_gamma_{gamma}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_accepted_sequence_sp_dy_gamma_{gamma}_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_1b_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_{gamma}_70b_7b_1b_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_accepted_sequence_sp_dy_gamma_{gamma}_70b_7b_1b_adv_topkp0_fp16_2048tok.csv',
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

dataset = "alpaca"
EXPERIMENTS_BL = ['70b']
EXPERIMENTS = ["70b_7b", "70b_7b_68m_dy", "70b_7b_68m_adv"]
# 創建一個 DataFrame 來存儲 baseline 的結果
gamma_range = range(2, 21)
baseline_df = pd.DataFrame(columns=gamma_range, index=EXPERIMENTS_BL)
# 創建一個 DataFrame 來存儲其他 experiment 的結果
experiment_df = pd.DataFrame(columns=gamma_range, index=EXPERIMENTS)


# 計算 baseline 的結果
for experiment in EXPERIMENTS_BL:
    sequence_time_path, sequence_length_path , accetped_seq_path = update_file_path(experiment, dataset,gamma=1)
    avg_value = 0
    if os.path.exists(sequence_time_path) and os.path.exists(sequence_length_path):
        with open(sequence_time_path, 'r') as f:
            time_values = [float(line.strip()) for line in f.readlines() if line.strip()]
        with open(sequence_length_path, 'r') as f:
            length_values = [float(line.strip()) for line in f.readlines() if line.strip()]
        
        if time_values and length_values:
            avg_value = np.sum(np.array(length_values)) / np.sum(np.array(time_values))
        for i in gamma_range:
            baseline_df.at[experiment, i] = avg_value  # 存入 DataFrame
    else:
        for i in gamma_range:
            baseline_df.at[experiment, i] = "N/A"


# 主程式
data=[]
hist=[]
ratios=[]
speeds=[]
# 計算其他 experiment 的結果並與 baseline 做對比
for experiment in EXPERIMENTS:
    for gamma in gamma_range:
        sequence_time_path, sequence_length_path, accetped_seq_path = update_file_path(experiment, dataset,gamma)
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
                bl_value = baseline_df.at[EXPERIMENTS_BL[0], gamma]
                if bl_value != "N/A" and bl_value != 0:
                    ratio = avg_value / float(bl_value)
                else:
                    ratio = "N/A"
                experiment_df.at[experiment, gamma] = round(avg_value, 2)
        else:
            experiment_df.at[experiment, gamma] = "N/A"






# 開始繪製圖表

# 設置不同的顏色和標記
colors = ['blue', 'green', 'red', 'black', 'black']
markers = ['o', 'x', '^', 's', '*']

labels_name={
    "70b_7b":"SD 70b-7b",
    "70b_7b_68m_dy":'Self/Pre-verify 70b_7b_68m', 
    "70b_7b_68m_adv":'HSDDW 70b_7b_68m'
}

# 假設 baseline_df 和 experiment_df 已經被填充了數據
plt.figure(figsize=(10, 6))

# 為每個 Baseline 實驗設置名稱並繪圖
for i, experiment in enumerate(EXPERIMENTS_BL):
    plt.plot(gamma_range, baseline_df.loc[experiment], 
             label=f"{experiment}",  # 為 baseline 命名
             color=colors[i], marker=markers[i])

# 為每個其他實驗設置名稱並繪圖
for i, experiment in enumerate(EXPERIMENTS):
    plt.plot(gamma_range, experiment_df.loc[experiment], 
             label=labels_name[experiment],  # 為其他實驗命名
             color=colors[i+1], marker=markers[i+1])




# 添加標籤和圖例
plt.xlabel('Window Size')
plt.ylabel('Speed (toks/sec)')
plt.title('Speed Comparison Across Different Window Sizes')

# 設置圖例，標題為 "Experiments"，圖例會顯示不同實驗的名稱
plt.legend(title="Experiments", loc="upper right", bbox_to_anchor=(1, 1))  # 圖例固定在右上角


# 顯示圖表
plt.show()
plt.xticks(gamma_range)  # 設置 X 軸顯示 gamma_range 內的整數




# 添加標題和標籤
plt.title('Speed Comparison and Dynamic Gamma')
plt.xlabel('Window Size', fontsize=14)  # 設定 X 軸標籤字體大小為 14
plt.ylabel('Speed (toks/sec)', fontsize=14)  # 設定 Y 軸標籤字體大小為 14


# 儲存圖像
output_file = f"/work/valex1377/LLMSpeculativeSampling/scripts/hir_sd_w_dy_wdw/figs/{dataset}_Speed_dynamic_and_baseline_gamma_plot.png"
plt.savefig(output_file)