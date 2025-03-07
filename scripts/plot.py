# # ============================================Accepted Entropy and Reject Entropy================

# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # 定義資料夾路徑
# base_dir = "/work/valex1377/LLMSpeculativeSampling/experiments"
# # 定義範圍
# i_range = range(2, 21)
# experiments = ["70b_68m", "70b_7b", "7b_68m"]

# # 初始化字典來儲存所有實驗的平均值
# accepted_averages_all = {}
# reject_averages_all = {}

# for experiment in experiments:
#     accepted_averages = {}
#     reject_averages = {}

#     for i in i_range:
#         # 構建 accepted 檔案的路徑
#         folder_name = f"sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok_record_only_old"
#         accepted_file_name = f"sp_dy_gamma_etp_accepted_entropy_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
#         accepted_file_path = os.path.join(base_dir, folder_name, accepted_file_name)

#         # 構建 reject 檔案的路徑
#         reject_file_name = f"sp_dy_gamma_etp_reject_entropy_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
#         reject_file_path = os.path.join(base_dir, folder_name, reject_file_name)

#         # 計算 accepted 檔案的平均值
#         if os.path.exists(accepted_file_path):
#             with open(accepted_file_path, 'r') as f:
#                 data = f.readlines()

#             values = [float(x) for line in data for x in line.strip().split(',') if x]

#             if values:
#                 avg_value = np.mean(values)
#                 accepted_averages[i] = avg_value
#             else:
#                 print(f"No valid data in accepted file: {accepted_file_path}")
#         else:
#             print(f"Accepted file not found: {accepted_file_path}")

#         # 計算 reject 檔案的平均值
#         if os.path.exists(reject_file_path):
#             with open(reject_file_path, 'r') as f:
#                 data = f.readlines()

#             values = [float(x) for line in data for x in line.strip().split(',') if x]

#             if values:
#                 avg_value = np.mean(values)
#                 reject_averages[i] = avg_value
#             else:
#                 print(f"No valid data in reject file: {reject_file_path}")
#         else:
#             print(f"Reject file not found: {reject_file_path}")

#     # 儲存每個實驗的平均值
#     accepted_averages_all[experiment] = accepted_averages
#     reject_averages_all[experiment] = reject_averages

# # 開始繪製圖表
# plt.figure(figsize=(10, 6))

# # 設置不同的顏色和標籤
# colors = ['blue', 'green', 'red']
# markers = ['o', 's', '^']

# # 繪製 accepted 和 reject 兩條線
# for idx, experiment in enumerate(experiments):
#     x_values = list(i_range)
#     accepted_y_values = [accepted_averages_all[experiment].get(i, np.nan) for i in i_range]
#     reject_y_values = [reject_averages_all[experiment].get(i, np.nan) for i in i_range]
    
#     # 繪製 accepted 平均值
#     plt.plot(x_values, accepted_y_values, label=f'Accepted Entropy ({experiment})', marker=markers[idx], color=colors[idx])
    
#     # 繪製 reject 平均值
#     plt.plot(x_values, reject_y_values, label=f'Reject Entropy ({experiment})', linestyle='--', marker=markers[idx], color=colors[idx])

# # 添加標題和標籤
# plt.title('Accepted vs Reject Entropy')
# plt.xlabel('Gamma')
# plt.ylabel('Average Entropy')
# plt.legend()

# # 儲存圖像
# output_file = "/work/valex1377/LLMSpeculativeSampling/Entropy_vs_Gamma.png"
# plt.savefig(output_file)

# # 顯示圖像
# plt.show()

# print(f"Plot saved as {output_file}")








# #===================================Accepted Rate===========================


# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # 定義資料夾路徑
# base_dir = "/work/valex1377/LLMSpeculativeSampling/experiments"
# # 定義範圍
# i_range = range(2, 21)
# experiments = ["70b_68m", "70b_7b", "7b_68m"]

# # 初始化用於存儲不同 i 的平均值
# experiment_averages = {experiment: [] for experiment in experiments}

# for experiment in experiments:
#     for i in i_range:
#         # 構建 sequence 檔案的路徑
#         folder_name = f"sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok_record_only"
#         sequence_file_name = f"sp_dy_gamma_etp_accepted_sequence_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
#         sequence_file_path = os.path.join(base_dir, folder_name, sequence_file_name)

#         # 計算 sequence 檔案的每個 value / i 並求平均值
#         if os.path.exists(sequence_file_path):
#             with open(sequence_file_path, 'r') as f:
#                 data = f.readlines()
            
#             values = []
#             for line in data:
#                 # 取出每行的數值，並將每個值除以 i
#                 values_in_line = [float(x) / i for x in line.strip().split(',') if x]
#                 values.extend(values_in_line)
            
#             # 計算當前 i 的平均值並加入對應的列表
#             if values:  # 確保有有效數據
#                 avg_value = np.mean(values)
#                 experiment_averages[experiment].append(avg_value)
#             else:
#                 experiment_averages[experiment].append(np.nan)  # 若無數據則填入 NaN 以便後續處理
#         else:
#             print(f"Sequence file not found: {sequence_file_path}")
#             experiment_averages[experiment].append(np.nan)

# # 開始繪製圖表
# plt.figure(figsize=(10, 6))

# # 繪製每個 experiment 的平均值隨著 i 變化的折線圖
# for experiment in experiments:
#     plt.plot(i_range, experiment_averages[experiment], label=f'{experiment}', marker='o')

# # 添加標題和標籤
# plt.title('Accepted Rate vs Different Gamma')
# plt.xlabel('gamma')
# plt.ylabel('Accepted Rate')
# plt.legend()

# # 儲存圖像
# output_file = "/work/valex1377/LLMSpeculativeSampling/accepted_rate_averages_plot.png"
# plt.savefig(output_file)

# # 顯示圖像
# plt.show()

# print(f"Plot saved as {output_file}")







####======================================dynamic_gamma_experiment_comparison_plot================
## Time vs gamma among different experiments



import os
import numpy as np
import matplotlib.pyplot as plt


# 定義範圍
i_range = range(2, 21)
# dataset_name = "mt_bench"
# dataset_name = "humaneval"
# dataset_name = "gsm8k"
dataset_name = "alpaca"
experiments_baseline = ['70b','7b','1b']
# experiments = ["70b_7b","70b_7b_68m_adv"]
experiments = ["70b_68m", "70b_7b", "7b_68m","70b_7b_68m","70b_7b_68m_adv"]
# experiments = ["70b_7b", "7b_1b","7b_68m","70b_7b_68m","70b_7b_68m_adv"]

# 初始化一個字典來儲存每個 experiment 的平均值
time_averages = {}
dynamic_time_averages = {}  # 儲存 dynamic gamma 的結果
# 定義資料夾路徑
base_dir = os.path.join("/work/valex1377/LLMSpeculativeSampling/experiments" ,dataset_name)

for experiment in experiments:
    averages = {}  # 儲存當前 experiment 的平均值
    dynamic_averages = {}  # 儲存當前 experiment 在 dynamic gamma 下的平均值

    for i in i_range:
        # 構建標準實驗資料夾和檔案路徑
        folder_name = f"sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok_record_only"
        sequence_length_name = f"sp_dy_gamma_etp_output_length_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
        sequence_time_name = f"sp_dy_gamma_etp_time_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
        sequence_length_path = os.path.join(base_dir, folder_name, sequence_length_name)
        sequence_time_path = os.path.join(base_dir, folder_name, sequence_time_name)    
        # 檢查標準實驗檔案是否存在並計算平均值
        if experiment not in ['70b_7b_68m','70b_7b_68m_adv'] :
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
                    averages[i] = avg_value
            else:
                if os.path.exists(sequence_time_path) is False:
                    print(f"No valid data in file: {sequence_time_path.split(f'/{dataset_name}/')[-1]}")
                elif os.path.exists(sequence_length_path) is False:
                    print(f"No valid data in file: {sequence_length_path.split(f'/{dataset_name}/')[-1]}")



        # 構建 dynamic gamma 資料夾和檔案路徑
        dynamic_folder_name = f"sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok"
        if experiment in ['70b_7b_68m','70b_7b_68m_adv'] :
            if experiment == '70b_7b_68m' :
                dynamic_time_name = f"sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
                dynamic_length_name = f"sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
            elif experiment == '70b_7b_68m_adv' :
                dynamic_time_name = f"sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
                dynamic_length_name = f"sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
            
        else:
            dynamic_time_name = f"sp_dy_gamma_etp_time_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
            dynamic_length_name = f"sp_dy_gamma_etp_output_length_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
        dynamic_time_path = os.path.join(base_dir, dynamic_folder_name, dynamic_time_name)
        dynamic_length_path = os.path.join(base_dir, dynamic_folder_name, dynamic_length_name)
        # 檢查 dynamic gamma 實驗檔案是否存在並計算平均值
        if os.path.exists(dynamic_time_path) or os.path.exists(dynamic_length_path):
            with open(dynamic_time_path, 'r') as f:
                data = f.readlines()
            dynamic_values = [float(line.strip()) for line in data if line.strip()]
            with open(dynamic_length_path, 'r') as f:
                length_data = f.readlines()
            dynamic_length_values = [float(line.strip()) for line in length_data if line.strip()]            
            if dynamic_values:
                tot_time_array = np.array(dynamic_length_values)/np.array(dynamic_values)
                dynamic_avg_value = np.sum(np.array(dynamic_length_values)) / np.sum(tot_time_array)
                dynamic_averages[i] = dynamic_avg_value
        else:
            if os.path.exists(dynamic_time_path) is False:
                print(f"No valid data in file: {dynamic_time_path.split(f'/{dataset_name}/')[-1]}")
            elif os.path.exists(dynamic_length_path) is False:
                print(f"No valid data in file: {dynamic_length_path.split(f'/{dataset_name}/')[-1]}")   
            
    # 儲存實驗結果
    time_averages[experiment] = averages
    dynamic_time_averages[experiment] = dynamic_averages



baseline_time_averages = {}  # 儲存當前 experiment 的平均值
for idx, experiment in enumerate(experiments_baseline):


    folder_name = f"{experiment}_fp16_2048tok_record_only"
    sequence_length_name = f"entropy_mesure_output_length_{experiment}_fp16_2048tok_record_only.csv"
    sequence_time_name = f"entropy_mesure_time_{experiment}_fp16_2048tok_record_only.csv"
    sequence_length_path = os.path.join(base_dir, folder_name, sequence_length_name)
    sequence_time_path = os.path.join(base_dir, folder_name, sequence_time_name)    
    
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


# 開始繪製圖表
plt.figure(figsize=(10, 6))

# 設置不同的顏色和標記
colors = ['blue', 'green', 'red', 'black', 'black']
markers = ['o', 's', '^', 'x', '*']

# 繪製各個 experiment 的時間平均
for idx, experiment in enumerate(experiments):
       
    # if None in dynamic_time_averages[experiment].values():
    #     continue
    averages = time_averages[experiment]
    dynamic_averages = dynamic_time_averages[experiment]
    

    x_values = sorted(averages.keys())  # i 值
    y_values = [averages[i] for i in x_values]  # 標準實驗的平均值
    dynamic_x_values = sorted(dynamic_averages.keys())  # i 值

    if experiment not in ['70b_7b_68m','70b_7b_68m_adv'] :
        dynamic_y_values = [dynamic_averages.get(i, np.nan) for i in x_values]  # dynamic gamma 的平均值
    else:
        dynamic_y_values = [dynamic_averages.get(i, np.nan) for i in dynamic_x_values]  # dynamic gamma 的平均值
    if experiment not in ['70b_7b_68m','70b_7b_68m_adv'] :
        plt.plot(x_values, y_values, label=f'SD ({experiment})', marker=markers[idx], color=colors[idx])
        if np.isfinite(dynamic_y_values).all():
            plt.plot(x_values, dynamic_y_values, label=f'Dynamic Gamma SD ({experiment})', marker=markers[idx], linestyle='--', color=colors[idx])
    else:
        plt.plot(dynamic_x_values, dynamic_y_values, label=f'Dynamic Gamma SD ({experiment})', marker=markers[idx], linestyle='--', color=colors[idx])        

    if experiment not in  ['70b_7b_68m','70b_7b_68m_adv'] :
        if np.isfinite(dynamic_y_values).all():
            # 標記標準實驗的最大值
            if len(y_values) > 0:
                max_y = max(y_values)
                max_x = x_values[np.argmax(y_values)]
                plt.text(max_x, max_y, f'{max_y:.2f}', ha='center', va='bottom', fontsize=10, color=colors[idx])
                # 標記 dynamic gamma 的最大值
                max_dynamic_y = max(dynamic_y_values)
                max_dynamic_x = x_values[np.argmax(dynamic_y_values)]
                plt.text(max_dynamic_x, max_dynamic_y, f'{max_dynamic_y:.2f}', ha='center', va='bottom', fontsize=10, color=colors[idx])
    else:
        # 標記 dynamic gamma 的最大值
        if np.isfinite(dynamic_y_values).all():
            max_dynamic_y = max(dynamic_y_values)
            max_dynamic_x = dynamic_x_values[np.argmax(dynamic_y_values)]
            plt.text(max_dynamic_x, max_dynamic_y, f'{max_dynamic_y:.2f}', ha='center', va='bottom', fontsize=10, color=colors[idx])
        # pass        
    
for idx, experiment in enumerate(experiments_baseline):
    # if baseline_time_averages[experiment] is not None:
    averages = baseline_time_averages[experiment]
    y_values = [averages]*len(dynamic_x_values)
    
    plt.plot(dynamic_x_values, y_values, label=f'Baseline Time Average ({experiment})', marker=markers[idx], linestyle='--', color='pink')



# 添加標題和標籤
plt.title('Speed Comparison and Dynamic Gamma')
plt.xlabel('Gamma')
plt.ylabel('Speed (toks/sec)')
plt.legend()

# 儲存圖像
output_file = f"/work/valex1377/LLMSpeculativeSampling/{dataset_name}_Speed_dynamic_and_baseline_gamma_plot.png"
plt.savefig(output_file)

# 顯示圖像
plt.show()

print(f"Plot saved as {output_file}")





