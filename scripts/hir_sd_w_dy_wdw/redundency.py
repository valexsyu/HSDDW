
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd

GG=10
GG1=10
bins = 10  # 根據範圍0~i設置bins的數量
def update_file_path(experiment, dataset) :

    ROOT="/home/valexsyu/Documents/battleship/speculative_decoding/"
    LENGTH_FILE_NAME={
        '70b':  ROOT+f'experiments/{dataset}/llama-2-70b-chat_topk0p0_fp16_2048tok/length_output_tokens.csv',
        '68m':  ROOT+f'experiments/{dataset}/68m_fp16_2048tok_record_only/entropy_mesure_output_length_68m_fp16_2048tok_record_only.csv',
        # '70b_7b' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_1b' : ROOT + f'new_experiments/{dataset}/10_70b_1b_topkp0_fp16_2048tok/sp_output_length_10_70b_1b_topkp0_fp16_2048tok.csv',
        '70b_68m' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok.csv',
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
        '70b_1b' : ROOT + f'new_experiments/{dataset}/10_70b_1b_topkp0_fp16_2048tok/sp_time_10_70b_1b_topkp0_fp16_2048tok.csv',
        '70b_68m' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_time_sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok.csv',
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
        '70b_1b' : ROOT + f'new_experiments/{dataset}/10_70b_1b_topkp0_fp16_2048tok/sp_accepted_sequence_10_70b_1b_topkp0_fp16_2048tok.csv',
        '70b_68m' :ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_accepted_sequence_sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_accepted_sequence_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_68m': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_accepted_sequence_sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_accepted_sequence_sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_accepted_sequence_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_1b_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_accepted_sequence_sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok.csv',
    }    
    
    GAMMA_FILE_NAME={
        '70b':  ROOT+f'experiments/{dataset}/llama-2-70b-chat_topk0p0_fp16_2048tok/time.csv',
        '68m':  ROOT+f'experiments/{dataset}/68m_fp16_2048tok_record_only/entropy_mesure_gamma_sequence_68m_fp16_2048tok_record_only.csv',
        # '70b_7b': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_gamma_sequence_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_gamma_sequence_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_1b' : ROOT + f'new_experiments/{dataset}/10_70b_1b_topkp0_fp16_2048tok/sp_gamma_sequence_10_70b_1b_topkp0_fp16_2048tok.csv',
        '70b_68m' :ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_gamma_sequence_sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_gamma_sequence_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_68m': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_gamma_sequence_sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_gamma_sequence_sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_gamma_sequence_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_1b_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_gamma_sequence_sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok.csv',
    }      
    return TIME_FILE_NAME[experiment], LENGTH_FILE_NAME[experiment] , ACCEPTED_FILE_NAME[experiment], GAMMA_FILE_NAME[experiment]




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
EXPERIMENTS_BL = []
EXPERIMENTS = ["70b_7b", "70b_68m","70b_7b_68m_dy","70b_7b_68m_adv"]
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


all_redundant={}
all_target_rate={}
only_math_speed={}
# 計算 baseline 的結果
for experiment in EXPERIMENTS_BL:

    sequence_time_path, sequence_length_path , accepted_seq_path, gamma_seq_path = update_file_path(experiment, dataset)
    redundant = 0
    if os.path.exists(accepted_seq_path) and os.path.exists(sequence_length_path) and os.path.exists(gamma_seq_path):
        with open(gamma_seq_path, 'r') as f:
            gamma_values = [float(line.strip()) for line in f.readlines() if line.strip()]
        with open(sequence_length_path, 'r') as f:
            length_values = [float(line.strip()) for line in f.readlines() if line.strip()]
        with open(accepted_seq_path, 'r') as f:
            accepted_values = [float(line.strip()) for line in f.readlines() if line.strip()]
            
        redundant = np.sum(np.array(accepted_values)) / np.sum(np.array(gamma_values))
        target_rate = len(gamma_values) / np.sum(np.array(length_values))
        all_redundant[experiment]=redundant
        all_target_rate[experiment] = target_rate
        
    else:
        breakpoint()
                      
                       


# 主程式
data=[]
hist=[]
ratios={}

# 計算其他 experiment 的結果並與 baseline 做對比
for experiment in EXPERIMENTS:
    
    sequence_time_path, sequence_length_path , accepted_seq_path, gamma_seq_path = update_file_path(experiment, dataset)
    redundant = 0
    if os.path.exists(accepted_seq_path) and os.path.exists(sequence_length_path) and os.path.exists(gamma_seq_path):
        with open(gamma_seq_path, 'r') as f:
            data = f.readlines()
            gamma_values=[]
            for i,line in enumerate(data):
                gamma_values.extend([float(x) for x in line.strip().split(',') if x])
        with open(sequence_length_path, 'r') as f:
            length_values = [float(line.strip()) for line in f.readlines() if line.strip()]
        with open(accepted_seq_path, 'r') as f:
            data = f.readlines()
            accepted_values=[]
            for i,line in enumerate(data):
                accepted_values.extend([float(x) for x in line.strip().split(',') if x])            

        redundant = np.sum(np.array(accepted_values)) / np.sum(np.array(gamma_values))
        target_rate = len(gamma_values) / np.sum(np.array(length_values))
        all_redundant[experiment]=str(round((1-redundant)*100,2))+'%'
        all_target_rate[experiment] = str(round(target_rate*100,2))+'%'
    else:
        breakpoint()        


print(f"all_redundant:{all_redundant}")
print(f"all_target_rate:{all_target_rate}")

