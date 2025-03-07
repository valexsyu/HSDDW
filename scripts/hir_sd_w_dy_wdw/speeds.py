

####======================================dynamic_gamma_experiment_comparison_plot================
## Time vs gamma among different experiments



import os
import numpy as np
import matplotlib.pyplot as plt
import csv

# 定義範圍
DATASETS = ["mt_bench" , "humaneval" ,"gsm8k", "alpaca"]
# DATASETS = ["mt_bench"]
EXPERIMENTS_BL = ['70b']
# EXPERIMENTS = ["70b_7b", "70b_1b", "70b_68m","70b_7b_68m"]
# EXPERIMENTS = ["70b_7b", "70b_1b", "70b_68m"]
EXPERIMENTS = ["70b_7b", "70b_7b_68m"]


# FILE_NAME_SEQ_LEN_BASE =(
#     f"""entropy_mesure_output_length_{{experiment}}_fp16_2048tok_record_only.csv"""
# )
# FILE_NAME_SEQ_LEN = (
#     f"""sp_output_length_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
# )

# FILE_NAME_TIME_BASE =(
#     f"""entropy_mesure_time_{{experiment}}_fp16_2048tok_record_only.csv"""
# )
# FILE_NAME_TIME = (
#     f"""sp_time_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
# )


# LENGTH_FILE_NAME={
#     '70b':  FILE_NAME_SEQ_LEN_BASE,
#     '70b_7b': FILE_NAME_SEQ_LEN ,
#     '70b_1b': FILE_NAME_SEQ_LEN ,
#     '70b_68m': FILE_NAME_SEQ_LEN
# }
# TIME_FILE_NAME={
#     '70b':  FILE_NAME_TIME_BASE,
#     '70b_7b': FILE_NAME_TIME ,
#     '70b_1b': FILE_NAME_TIME ,
#     '70b_68m': FILE_NAME_TIME
# }

# def update_experiment_paths(dataset_name):
#     return {
#         '70b'    : f'/work/valex1377/LLMSpeculativeSampling/experiments/{dataset_name}/70b_fp16_2048tok_record_only',
#         '70b_7b' : f'/work/valex1377/LLMSpeculativeSampling/new_experiments/{dataset_name}/10_70b_7b_topkp0_fp16_2048tok',
#         '70b_1b' : f'/work/valex1377/LLMSpeculativeSampling/new_experiments/{dataset_name}/10_70b_1b_topkp0_fp16_2048tok',
#         '70b_68m': f'/work/valex1377/LLMSpeculativeSampling/new_experiments/{dataset_name}/10_70b_68m_topkp0_fp16_2048tok'
#     }






FILE_NAME_SEQ_LEN_BASE =(
    f"""entropy_mesure_output_length_{{experiment}}_fp16_2048tok_record_only.csv"""
)
FILE_NAME_SEQ_LEN = (
    f"""sp_dy_gamma_etp_output_length_sp_dy_gamma_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
)
FILE_NAME_SEQ_LEN_HRCHL = (
    f"""sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
)
FILE_NAME_SEQ_LEN_HRCHL_ADV = (
    f"""sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
)



FILE_NAME_TIME_BASE =(
    f"""entropy_mesure_time_{{experiment}}_fp16_2048tok_record_only.csv"""
)
FILE_NAME_TIME = (
    f"""sp_dy_gamma_etp_time_sp_dy_gamma_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
)
FILE_NAME_TIME_HRCHL = (
    f"""sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
)
FILE_NAME_TIME_HRCHL_ADV = (
    f"""sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_10_{{experiment}}_topkp0_fp16_2048tok.csv"""
)

LENGTH_FILE_NAME={
    '70b':  FILE_NAME_SEQ_LEN_BASE,
    '70b_7b': FILE_NAME_SEQ_LEN ,
    '70b_1b': FILE_NAME_SEQ_LEN ,
    '70b_68m': FILE_NAME_SEQ_LEN,
    '70b_7b_68m': FILE_NAME_SEQ_LEN_HRCHL ,
    '70b_7b_68m_adv' : FILE_NAME_SEQ_LEN_HRCHL_ADV,
    '70b_7b_1b_adv' : FILE_NAME_SEQ_LEN_HRCHL_ADV,
}
TIME_FILE_NAME={
    '70b':  FILE_NAME_TIME_BASE,
    '70b_7b': FILE_NAME_TIME ,
    '70b_1b': FILE_NAME_TIME ,
    '70b_68m': FILE_NAME_TIME,
    '70b_7b_68m': FILE_NAME_TIME_HRCHL ,
    '70b_7b_68m_adv' : FILE_NAME_TIME_HRCHL_ADV,
    '70b_7b_1b_adv' : FILE_NAME_TIME_HRCHL_ADV,
}

def update_experiment_paths(dataset_name):
    return {
        '70b'    : f'/work/valex1377/LLMSpeculativeSampling/experiments/{dataset_name}/70b_fp16_2048tok_record_only',
        '70b_7b' : f'/work/valex1377/LLMSpeculativeSampling/experiments/{dataset_name}/sp_dy_gamma_10_70b_7b_topkp0_fp16_2048tok_record_only',
        '70b_1b' : f'/work/valex1377/LLMSpeculativeSampling/experiments/{dataset_name}/sp_dy_gamma_10_70b_1b_topkp0_fp16_2048tok_record_only',
        '70b_68m': f'/work/valex1377/LLMSpeculativeSampling/experiments/{dataset_name}/sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok_record_only',
        '70b_7b_68m' : f'/work/valex1377/LLMSpeculativeSampling/experiments/{dataset_name}/sp_dy_gamma_10_70b_7b_68m_topkp0_fp16_2048tok',
        '70b_7b_68m_adv' : f'/work/valex1377/LLMSpeculativeSampling/experiments/{dataset_name}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok',
        '70b_7b_1b_adv' : f'/work/valex1377/LLMSpeculativeSampling/experiments/{dataset_name}/sp_dy_gamma_10_70b_7b_1b_adv_topkp0_fp16_2048tok',
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



speed_averages = {}

for dataset in DATASETS :
    print(f"======={dataset}=======")
    experiments_folder_path = update_experiment_paths(dataset)
    baseline_time_averages = {}  # 儲存當前 experiment 的平均值
    for idx, experiment in enumerate(EXPERIMENTS_BL):

        avg_value = 0
        sequence_time_path = os.path.join(
            experiments_folder_path[experiment], 
            TIME_FILE_NAME[experiment].format(experiment=experiment),
        )    
        sequence_length_path = os.path.join(
            experiments_folder_path[experiment], 
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
            bl_time = avg_value        
                
        else:
            if os.path.exists(sequence_time_path) is False:
                print(f"No valid data in file: {sequence_time_path}")
            elif os.path.exists(sequence_length_path) is False:
                print(f"No valid data in file: {sequence_length_path}")           
            bl_time = None   
            continue
            # 儲存實驗結果    
            
        print(f"baseline_time_averages:{round(bl_time,3)}")
        for experiment in EXPERIMENTS:
            averages = {}  # 儲存當前 experiment 的平均值
            dynamic_averages = {}  # 儲存當前 experiment 在 dynamic gamma 下的平均值
            avg_value = 0
            sequence_time_path = os.path.join(
                experiments_folder_path[experiment], 
                TIME_FILE_NAME[experiment].format(experiment=experiment),
            )    
            sequence_length_path = os.path.join(
                experiments_folder_path[experiment], 
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
                avg_value=0
                if time_values or length_values:
                    tot_time_array = np.array(length_values)/np.array(time_values) 
                    avg_value = np.sum(np.array(length_values))/np.sum(tot_time_array)  # 計算平均值


            
                
                
    
                ratio=round(avg_value/bl_time, 3)
                print(f"{experiment}: tokens/sec:{round(avg_value,3)} ; Ratio:{ratio} ")        
        
            else:
                if os.path.exists(sequence_time_path) is False:
                    print(f"No valid data in file: {sequence_time_path}")
                elif os.path.exists(sequence_length_path) is False:
                    print(f"No valid data in file: {sequence_length_path}")
                continue
                

            # 儲存實驗結果
            speed_averages[experiment] = avg_value






        # models = list(speed_averages.keys())
        # values = [speed_averages[model] for model in models]
        # baseline_model = list(baseline_time_averages.keys())[0]
        # baseline_value = baseline_time_averages[baseline_model]  
        # print(f"======={dataset}=======")
        # print(f"baseline_time_averages:{round(baseline_value,3)}")
        # for j, experiment in enumerate(EXPERIMENTS):    
        #     ratio=round(values[j]/baseline_value, 3)
        #     speed=round(speed_averages[experiment]/baseline_value, 3)
            
        
        #     print(f"{experiment}: tokens/sec:{round(speed_averages[experiment],3)} ; Ratio:{ratio} ")
        



