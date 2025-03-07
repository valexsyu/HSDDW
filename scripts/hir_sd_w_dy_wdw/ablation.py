

# # ####======================================dynamic_gamma_experiment_comparison_plot================
# # ## Time vs gamma among different experiments



# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import csv

# ##===========main result==========

# # # 定義範圍

# # # DATASETS = ["mt_bench"]
# # EXPERIMENTS_BL = ['70b']
# # EXPERIMENTS = ["70b_7b", "70b_7b_68m_adv"]

# ##=============ablation================
# # 定義範圍
# # DATASETS = ["mt_bench"]
# DATASETS = ["mt_bench" , "humaneval" ,"gsm8k", "alpaca"]
# EXPERIMENTS_BL = ['70b']
# EXPERIMENTS = ["70b_7b", "70b_7b_68m", "70b_7b_dy","70b_7b_68m_dy", "70b_7b_68m_adv"]



# # # FILE_NAME_SEQ_LEN_BASE =(
# # #     f"""entropy_mesure_output_length_{{experiment}}_fp16_2048tok_record_only.csv"""
# # # )
# # # FILE_NAME_SEQ_LEN = (
# # #     f"""sp_output_length_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
# # # )

# # # FILE_NAME_TIME_BASE =(
# # #     f"""entropy_mesure_time_{{experiment}}_fp16_2048tok_record_only.csv"""
# # # )
# # # FILE_NAME_TIME = (
# # #     f"""sp_time_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
# # # )


# # # LENGTH_FILE_NAME={
# # #     '70b':  FILE_NAME_SEQ_LEN_BASE,
# # #     '70b_7b': FILE_NAME_SEQ_LEN ,
# # #     '70b_1b': FILE_NAME_SEQ_LEN ,
# # #     '70b_68m': FILE_NAME_SEQ_LEN
# # # }
# # # TIME_FILE_NAME={
# # #     '70b':  FILE_NAME_TIME_BASE,
# # #     '70b_7b': FILE_NAME_TIME ,
# # #     '70b_1b': FILE_NAME_TIME ,
# # #     '70b_68m': FILE_NAME_TIME
# # # }

# # # def update_experiment_paths(dataset_name):
# # #     return {
# # #         '70b'    : f'/work/valex1377/LLMSpeculativeSampling/experiments/{dataset_name}/70b_fp16_2048tok_record_only',
# # #         '70b_7b' : f'/work/valex1377/LLMSpeculativeSampling/new_experiments/{dataset_name}/10_70b_7b_topkp0_fp16_2048tok',
# # #         '70b_1b' : f'/work/valex1377/LLMSpeculativeSampling/new_experiments/{dataset_name}/10_70b_1b_topkp0_fp16_2048tok',
# # #         '70b_68m': f'/work/valex1377/LLMSpeculativeSampling/new_experiments/{dataset_name}/10_70b_68m_topkp0_fp16_2048tok'
# # #     }






# # FILE_NAME_SEQ_LEN_BASE =(
# #     f"""entropy_mesure_output_length_{{experiment}}_fp16_2048tok_record_only.csv"""
# # )
# # FILE_NAME_SEQ_LEN = (
# #     f"""sp_dy_gamma_etp_output_length_sp_dy_gamma_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
# # )
# # FILE_NAME_SEQ_LEN_HRCHL = (
# #     f"""sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
# # )
# # FILE_NAME_SEQ_LEN_HRCHL_ADV = (
# #     f"""sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
# # )



# # FILE_NAME_TIME_BASE =(
# #     f"""entropy_mesure_time_{{experiment}}_fp16_2048tok_record_only.csv"""
# # )
# # FILE_NAME_TIME = (
# #     f"""sp_dy_gamma_etp_time_sp_dy_gamma_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
# # )
# # FILE_NAME_TIME_HRCHL = (
# #     f"""sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
# # )
# # FILE_NAME_TIME_HRCHL_ADV = (
# #     f"""sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
# # )

GG=10
GG1=10

def update_file_path(experiment, dataset) :
    # ROOT='/home/valexsyu/Documents/battleship/speculative_decoding/'
    ROOT="/work/valex1377/LLMSpeculativeSampling/"
    LENGTH_FILE_NAME={
        '70b':  ROOT+f'experiments/{dataset}/llama-2-70b-chat_topk0p0_fp16_2048tok/length_output_tokens.csv',
        '70b_7b' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_temp0' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_temp1e-20_fp16_2048tok_record_only_pure/sp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_temp1' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_temp1_fp16_2048tok_record_only_pure/sp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        # '70b_7b' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_68m': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        #'70b_7b_68m_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_temp1_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_adv_temp0' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_temp1e-20_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_1b_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok.csv',
    }
    TIME_FILE_NAME={
        '70b':  ROOT+f'experiments/{dataset}/llama-2-70b-chat_topk0p0_fp16_2048tok/time.csv',
        '70b_7b': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_temp0': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_temp1e-20_fp16_2048tok_record_only_pure/sp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_temp1': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_temp1_fp16_2048tok_record_only_pure/sp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        # '70b_7b': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only/sp_dy_gamma_etp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
        '70b_7b_68m': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_{GG}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_dy': ROOT + f'experiments/{dataset}/sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_{GG1}_70b_7b_68m_topkp0_fp16_2048tok.csv',
        #'70b_7b_68m_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_temp1_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_68m_adv_temp0' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_temp1e-20_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
        '70b_7b_1b_adv' : ROOT + f'experiments/{dataset}/sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok.csv',
    }
    return TIME_FILE_NAME[experiment], LENGTH_FILE_NAME[experiment]
# # elif DATASETS[0] == 'humaneval':

# # #     ROOT="/work/valex1377/LLMSpeculativeSampling/"
# # #     LENGTH_FILE_NAME={
# # #         '70b':  ROOT+'experiments/humaneval/llama-2-70b-chat_topk0p0_fp16_2048tok/length_output_tokens.csv',
# # #         '70b_7b' : ROOT + 'experiments/humaneval/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_dy': ROOT + 'experiments/humaneval/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m': ROOT +'experiments/humaneval/sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m_dy': ROOT +'experiments/humaneval/sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m_adv' : ROOT +'experiments/humaneval/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
# # #     }
# # #     TIME_FILE_NAME={
# # #         '70b':  ROOT+'experiments/humaneval/llama-2-70b-chat_topk0p0_fp16_2048tok/time.csv',
# # #         '70b_7b': ROOT + 'experiments/humaneval/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_dy': ROOT + 'experiments/humaneval/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m': ROOT + 'experiments/humaneval/sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m_dy': ROOT + 'experiments/humaneval/sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m_adv' : ROOT +'experiments/humaneval/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
# # #     }

# # # elif DATASETS[0] == 'gsm8k':

# # #     ROOT="/work/valex1377/LLMSpeculativeSampling/"
# # #     LENGTH_FILE_NAME={
# # #         '70b':  ROOT+'experiments/gsm8k/llama-2-70b-chat_topk0p0_fp16_2048tok/length_output_tokens.csv',
# # #         '70b_7b' : ROOT + 'experiments/gsm8k/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_dy': ROOT + 'experiments/gsm8k/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m': ROOT +'experiments/gsm8k/sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m_dy': ROOT +'experiments/gsm8k/sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m_adv' : ROOT +'experiments/gsm8k/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
# # #     }
# # #     TIME_FILE_NAME={
# # #         '70b':  ROOT+'experiments/gsm8k/llama-2-70b-chat_topk0p0_fp16_2048tok/time.csv',
# # #         '70b_7b': ROOT + 'experiments/gsm8k/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_dy': ROOT + 'experiments/gsm8k/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m': ROOT + 'experiments/gsm8k/sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m_dy': ROOT + 'experiments/gsm8k/sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m_adv' : ROOT +'experiments/gsm8k/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
# # #     }

# # # elif DATASETS[0] == 'alpaca':

# # #     ROOT="/work/valex1377/LLMSpeculativeSampling/"
# # #     LENGTH_FILE_NAME={
# # #         '70b':  ROOT+'experiments/alpaca/llama-2-70b-chat_topk0p0_fp16_2048tok/length_output_tokens.csv',
# # #         '70b_7b' : ROOT + 'experiments/alpaca/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_dy': ROOT + 'experiments/alpaca/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_output_length_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m': ROOT +'experiments/alpaca/sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m_dy': ROOT +'experiments/alpaca/sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m_adv' : ROOT +'experiments/alpaca/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
# # #     }
# # #     TIME_FILE_NAME={
# # #         '70b':  ROOT+'experiments/alpaca/llama-2-70b-chat_topk0p0_fp16_2048tok/time.csv',
# # #         '70b_7b': ROOT + 'experiments/alpaca/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_dy': ROOT + 'experiments/alpaca/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok/sp_dy_gamma_etp_time_sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m': ROOT + 'experiments/alpaca/sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok_record_only_pure/sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m_dy': ROOT + 'experiments/alpaca/sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_5_70b_7b_68m_topkp0_fp16_2048tok.csv',
# # #         '70b_7b_68m_adv' : ROOT +'experiments/alpaca/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv',
# # #     }



# 讀取CSV檔案並將數據存儲在列表中
def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            data.extend(map(int, row))
    return data

# 計算每個bin的總數量


# speed_averages = {}

# for dataset in DATASETS :
#     print(f"======={dataset}=======")
#     baseline_time_averages = {}  # 儲存當前 experiment 的平均值
#     for idx, experiment in enumerate(EXPERIMENTS_BL):
#         sequence_time_path ,  sequence_length_path = update_file_path(experiment, dataset)
#         avg_value = 0
#         if os.path.exists(sequence_time_path) and os.path.exists(sequence_length_path):
            
#             with open(sequence_time_path, 'r') as f:
#                 data = f.readlines()
#             time_values = [float(line.strip()) for line in data if line.strip()]  # 將資料轉換為浮點數列表
#             with open(sequence_length_path, 'r') as f:
#                 data = f.readlines()
#             length_values = [float(line.strip()) for line in data if line.strip()]  # 將資料轉換為浮點數列表
            
#             if time_values and length_values:
#                 avg_value = np.sum(np.array(length_values))/np.sum(np.array(time_values))  # 計算平均值
#             bl_time = avg_value        
#         else:
#             print(f"{experiment} - {dataset}: No file")
#             avg_value = "N/A"  # 如果文件不存在，將值設為 "N/A"
          
#         print(f"baseline_time_averages:{round(bl_time,3)}")
#         for experiment in EXPERIMENTS:
#             averages = {}  # 儲存當前 experiment 的平均值
#             avg_value = 0
#             sequence_time_path ,  sequence_length_path = update_file_path(experiment, dataset)
#             if os.path.exists(sequence_time_path) and os.path.exists(sequence_length_path):
#                 with open(sequence_time_path, 'r') as f:
#                     data = f.readlines()
#                 time_values = [float(line.strip()) for line in data if line.strip()]  # 將資料轉換為浮點數列表
#                 with open(sequence_length_path, 'r') as f:
#                     data = f.readlines()
#                 length_values = [float(line.strip()) for line in data if line.strip()]  # 將資料轉換為浮點數列表
#                 avg_value=0
#                 if time_values or length_values:
#                     tot_time_array = np.array(length_values)/np.array(time_values) 
#                     avg_value = np.sum(np.array(length_values))/np.sum(tot_time_array)  # 計算平均值
#                 ratio=round(avg_value/bl_time, 3)
#                 print(f"{experiment}: tokens/sec:{round(avg_value,3)} ; Ratio:{ratio} ")        
        
#             else:
#                 print(f"{experiment} - {dataset}: No file")
#                 avg_value = "N/A"  # 如果文件不存在，將值設為 "N/A"
#             # 儲存實驗結果
#             speed_averages[experiment] = avg_value

        


import os
import numpy as np
import pandas as pd

DATASETS = ["mt_bench", "humaneval", "gsm8k", "alpaca"]
EXPERIMENTS_BL = ['70b']
EXPERIMENTS = ["70b_7b", "70b_7b_temp0","70b_7b_temp1", "70b_7b_68m", "70b_7b_dy", "70b_7b_68m_dy", "70b_7b_68m_adv", "70b_7b_68m_adv_temp0"]
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
        sequence_time_path, sequence_length_path = update_file_path(experiment, dataset)
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

# 計算其他 experiment 的結果並與 baseline 做對比
for experiment in EXPERIMENTS:
    for dataset in DATASETS:
        sequence_time_path, sequence_length_path = update_file_path(experiment, dataset)
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
        
        if experiment == '70b_7b' and experiment_df.at[experiment, dataset]!='N/A':
            if dataset == 'mt_bench' :    
                experiment_df.at['SpecDec++', dataset] = f"- (-)"
            else:
                experiment_df.at['SpecDec++', dataset] = f"- ({round(ratio*SPECDEC[dataset],2)}x)"
                



for dataset in DATASETS:
    if dataset == 'alpaca' :
        experiment_df.at['PERAL', dataset] = f"- (-)"
    else:
        experiment_df.at['PERAL', dataset] = f"- ({PERAL[dataset]}x)"

# 打印 Baseline 的結果
print("============================== Baseline Results =============================")
print(baseline_df.round(2))

# 打印其他 Experiment 的結果與 Baseline 的對比
print("\n======================== Experiment Results with Ratios =========================")
print(experiment_df.round(2))

