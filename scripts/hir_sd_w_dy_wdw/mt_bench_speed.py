
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
        '68m':  ROOT+f'experiments/{dataset}/68m_fp16_2048tok_record_only/entropy_mesure_output_length_68m_fp16_2048tok_record_only.csv',
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
        '68m':  ROOT+f'experiments/{dataset}/68m_fp16_2048tok_record_only/entropy_mesure_time_68m_fp16_2048tok_record_only.csv',
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
        '68m':  ROOT+f'experiments/{dataset}/68m_fp16_2048tok_record_only/entropy_mesure_accepted_sequence_68m_fp16_2048tok_record_only.csv',
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


dataset="mt_bench"
EXPERIMENTS_BL = ['70b']
EXPERIMENTS = ["70b_7b", "70b_68m"]
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
DATA_TYPES=['writing','roleplay','reasoning','math','coding','extraction','stem','humanities']

# 創建一個 DataFrame 來存儲 baseline 的結果
baseline_df = pd.DataFrame(columns=DATA_TYPES, index=EXPERIMENTS_BL)
# 創建一個 DataFrame 來存儲其他 experiment 的結果
experiment_df = pd.DataFrame(columns=DATA_TYPES, index=EXPERIMENTS)

baseline_length_df = pd.DataFrame(columns=DATA_TYPES, index=EXPERIMENTS_BL)
experiment_length_df = pd.DataFrame(columns=DATA_TYPES, index=EXPERIMENTS)


# 計算 baseline 的結果
for experiment in EXPERIMENTS_BL:

    sequence_time_path, sequence_length_path , accetped_seq_path = update_file_path(experiment, dataset)
    avg_value = 0
    if os.path.exists(sequence_time_path) and os.path.exists(sequence_length_path):
        with open(sequence_time_path, 'r') as f:
            time_values = [float(line.strip()) for line in f.readlines() if line.strip()]
        with open(sequence_length_path, 'r') as f:
            length_values = [float(line.strip()) for line in f.readlines() if line.strip()]
            j=0
        for i in range(0,len(time_values),10):
            avg_value = np.sum(np.array(length_values[i:i+10])) / np.sum(np.array(time_values[i:i+10]))
            baseline_length_df.at[experiment, DATA_TYPES[j]] = np.average(length_values[i:i+10])
            baseline_df.at[experiment, DATA_TYPES[j]] = avg_value
            j = j+1
                       


# 主程式
data=[]
hist=[]
ratios=[]
# 計算其他 experiment 的結果並與 baseline 做對比
for experiment in EXPERIMENTS:
    
    sequence_time_path, sequence_length_path, accetped_seq_path = update_file_path(experiment, dataset)
    avg_value = 0
    if os.path.exists(sequence_time_path) and os.path.exists(sequence_length_path):
        with open(sequence_time_path, 'r') as f:
            time_values = [float(line.strip()) for line in f.readlines() if line.strip()]
        with open(sequence_length_path, 'r') as f:
            length_values = [float(line.strip()) for line in f.readlines() if line.strip()]
        j=0
        for i in range(0,len(time_values),10):
            
            tot_time_array = np.array(length_values[i:i+10])/np.array(time_values[i:i+10]) 
            experiment_length_df.at[experiment, DATA_TYPES[j]] = np.average(length_values[i:i+10])
            avg_value = np.sum(np.array(length_values[i:i+10]))/np.sum(tot_time_array)  # 計算平均值                
            # 計算與 baseline 的比例
            experiment_df.at[experiment, DATA_TYPES[j]] = avg_value
            j = j+1



# # 假設 DATA_TYPES, baseline_df 和 experiment_df 已經正確定義並填充了數據
# DATA_TYPES = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']

# # 繪製 baseline_df 數據
# plt.plot(DATA_TYPES, baseline_df.loc['70b'], label='Baseline 70b', marker='o', color='blue')

# # 繪製 experiment_df 數據
# plt.plot(DATA_TYPES, experiment_df.loc['70b_7b'], label='Experiment 70b_7b', marker='s', color='green')
# plt.plot(DATA_TYPES, experiment_df.loc['70b_68m'], label='Experiment 70b_68m', marker='^', color='red')

# # 添加標籤和標題
# plt.xlabel('Data Types', fontsize=14)
# plt.ylabel('Speed', fontsize=14)  # 根據你的數據名稱可以修改這裡
# plt.title('Comparison of Speed by Data Type', fontsize=16)

# # 添加圖例
# plt.legend(loc="upper right")

# # 顯示圖表
# plt.xticks(rotation=45)  # 如果需要，可以將 X 軸標籤旋轉，以便更好地顯示
# plt.tight_layout()  # 調整布局以避免標籤被擋住

# output_file = "/work/valex1377/LLMSpeculativeSampling/scripts/hir_sd_w_dy_wdw/figs/mt_bench_speed.png"
# plt.savefig(output_file)
# print(f"Finish save plot : {output_file}")

# plt.close








# 對 baseline_df 進行最大值正規化
baseline_norm = baseline_df.loc['70b'] / baseline_df.loc['70b'].max()
# 對 experiment_length_df 的每一行進行最大值正規化
breakpoint()
experiment_norm_70b_7b = experiment_length_df.loc['70b_7b'] / experiment_length_df.loc['70b_7b'].max()
experiment_norm_70b_68m = experiment_length_df.loc['70b_68m'] / experiment_length_df.loc['70b_68m'].max()

# 繪製 baseline_df 正規化後的線
# plt.plot(DATA_TYPES, baseline_norm, label='Baseline 70b', marker='o', color='blue')

# 繪製 experiment_length_df 正規化後的線
plt.plot(DATA_TYPES, experiment_norm_70b_7b, label='Experiment 70b_7b', marker='s', color='green')
plt.plot(DATA_TYPES, experiment_norm_70b_68m, label='Experiment 70b_68m', marker='^', color='red')

# 添加標籤和標題
plt.xlabel('Data Types', fontsize=14)
plt.ylabel('Normalized Scores', fontsize=14)  # 正規化後的分數
plt.title('Normalized Comparison of Scores by Data Type', fontsize=16)

# 添加圖例
plt.legend(title="Experiments", loc="upper left")
output_file = "/work/valex1377/LLMSpeculativeSampling/scripts/hir_sd_w_dy_wdw/figs/mt_bench_speed_length.png"
# 顯示圖表
plt.xticks(rotation=45)  # 如果需要，可以將 X 軸標籤旋轉，以便更好地顯示
plt.tight_layout()  # 調整布局以避免標籤被擋住
plt.show()
plt.savefig(output_file)
print(f"Finish save plot : {output_file}")