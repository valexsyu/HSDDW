

####======================================dynamic_gamma_experiment_comparison_plot================
## Time vs gamma among different experiments



import os
import numpy as np
import matplotlib.pyplot as plt


# 定義範圍
i_range = range(10, 11)
dataset_name = "mt_bench"
# dataset_name = "humaneval"
experiments_baseline = ['70b']
experiments = ["70b_7b", "70b_68m"]
# experiments = ["70b_7b","70b_68m", "70b_7b_68m","70b_7b_68m_adv"]
# experiments = ["70b_7b", "70b_7b_68m","70b_7b_68m_adv"]

# 初始化一個字典來儲存每個 experiment 的平均值
speed_averages = {}
dynamic_time_averages = {}  # 儲存 dynamic gamma 的結果
# 定義資料夾路徑
base_dir = os.path.join("/work/valex1377/LLMSpeculativeSampling/experiments" ,dataset_name)



import csv
import matplotlib.pyplot as plt



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
    width = 0.8/num_experiment
    shift = 0.5
    models = list(speed_averages.keys())
    values = [list(speed_averages[model].values())[0] for model in models]
    baseline_model = list(baseline_time_averages.keys())[0]
    baseline_value = baseline_time_averages[baseline_model]    

    aligns=['center','center','center','center']
    for j, experiment in enumerate(experiments):    
        x = np.arange(bins + 1)
        speed=round(values[j]/baseline_value, 3)
        ax1.bar(x+(j-shift)*width, hist[j], width, label=experiment+f"({speed})"+"x", align=aligns[j])
    ax1.set_xlabel('Accepted Number Per Iteration',fontsize=14)
    ax1.set_ylabel('Totla Count',fontsize=14)
    ax1.set_title('Speculative Decoding with Different Model Sizes',fontsize=16)
    ax1.legend(fontsize=14)


    # 儲存圖像
    output_file = f"/work/valex1377/LLMSpeculativeSampling/paper_figs/1_motivation.png"
    plt.savefig(output_file)




# # 繪製直方圖（對數縮放）
# def plot_histogram(hist, bins, experiments, speed_averages, baseline_time_averages):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     num_experiment = len(experiments)
#     width = 0.8 / num_experiment
#     shift = 0.5
#     models = list(speed_averages.keys())
#     values = [list(speed_averages[model].values())[0] for model in models]
#     baseline_model = list(baseline_time_averages.keys())[0]
#     baseline_value = baseline_time_averages[baseline_model]

#     aligns = ['center', 'center', 'center', 'center']
#     for j, experiment in enumerate(experiments):
#         x = np.arange(bins + 1)
#         speed = round(values[j] / baseline_value, 3)
#         ax.bar(x + (j - shift) * width, hist[j], width, label=experiment + f"({speed})" + "x", align=aligns[j])

#     ax.set_xlabel('Accepted Number Per Iteration', fontsize=14)
#     ax.set_ylabel('Total Count (Log Scale)', fontsize=14)
#     ax.set_title('Speculative Decoding with Different Model Sizes (Log Scale)', fontsize=16)
#     ax.legend(fontsize=14)
#     ax.set_yscale('log')  # 设置 y 轴为对数尺度

#     # 儲存圖像
#     output_file = f"/work/valex1377/LLMSpeculativeSampling/paper_figs/1_motivation_log.png"
#     plt.savefig(output_file)



# # 繪製直方圖（次 y 轴）
# def plot_histogram(hist, bins, experiments, speed_averages, baseline_time_averages):
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#     ax2 = ax1.twinx()  # 创建次 y 轴
#     num_experiment = len(experiments)
#     width = 0.8 / num_experiment
#     shift = 0.5
#     models = list(speed_averages.keys())
#     values = [list(speed_averages[model].values())[0] for model in models]
#     baseline_model = list(baseline_time_averages.keys())[0]
#     baseline_value = baseline_time_averages[baseline_model]

#     aligns = ['center', 'center', 'center', 'center']
#     for j, experiment in enumerate(experiments):
#         x = np.arange(bins + 1)
#         speed = round(values[j] / baseline_value, 3)
#         if max(hist[j]) > 5000:
#             ax2.bar(x + (j - shift) * width, hist[j], width, label=experiment + f"({speed})" + "x", align=aligns[j], alpha=0.6)
#         else:
#             ax1.bar(x + (j - shift) * width, hist[j], width, label=experiment + f"({speed})" + "x", align=aligns[j])

#     ax1.set_xlabel('Accepted Number Per Iteration', fontsize=14)
#     ax1.set_ylabel('Total Count (Normal)', fontsize=14)
#     ax2.set_ylabel('Total Count (Large)', fontsize=14)
#     ax1.set_title('Speculative Decoding with Different Model Sizes', fontsize=16)
#     ax1.legend(fontsize=14)

#     # 儲存圖像
#     output_file = f"/work/valex1377/LLMSpeculativeSampling/paper_figs/1_motivation_secondary_axis.png"
#     plt.savefig(output_file)


# 主程式
data=[]
hist=[]
for j, experiment in enumerate(experiments):
    for i in i_range:
        # 構建標準實驗資料夾和檔案路徑
        if experiment == '70b_7b_68m' :
            folder_name = f"sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok"   
            file_name = f"sp_dy_gamma_etp_hrchl_accepted_sequence_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"  
            i = 5
            
        elif experiment == "70b_7b_68m_adv" :
            folder_name = f"sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok" 
            file_name = f"sp_dy_gamma_etp_hrchl_adv_accepted_sequence_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv" 
            i = 10  
        else:
            folder_name = f"sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok_record_only" 
            file_name = f"sp_dy_gamma_etp_accepted_sequence_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"   
        file_path = os.path.join(base_dir, folder_name, file_name)    
        bins = 10  # 根據範圍0~i設置bins的數量

    print(experiment)



    averages = {}  # 儲存當前 experiment 的平均值
    dynamic_averages = {}  # 儲存當前 experiment 在 dynamic gamma 下的平均值
    data.append(read_csv(file_path))

    hist.append(calculate_histogram(data[j], bins))


for experiment in experiments:
    averages = {}  # 儲存當前 experiment 的平均值
    dynamic_averages = {}  # 儲存當前 experiment 在 dynamic gamma 下的平均值

    for i in i_range:
        # 構建標準實驗資料夾和檔案路徑
        if experiment == '70b_7b_68m' :
            folder_name = f"sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok"
            sequence_length_name = f"sp_dy_gamma_etp_hrchl_output_length_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
            sequence_time_name = f"sp_dy_gamma_etp_hrchl_time_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
            sequence_length_path = os.path.join(base_dir, folder_name, sequence_length_name)
            sequence_time_path = os.path.join(base_dir, folder_name, sequence_time_name)   
            i = 5
            
        elif experiment == "70b_7b_68m_adv" :
            folder_name = f"sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok"
            sequence_length_name = f"sp_dy_gamma_etp_hrchl_adv_output_length_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
            sequence_time_name = f"sp_dy_gamma_etp_hrchl_adv_time_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
            sequence_length_path = os.path.join(base_dir, folder_name, sequence_length_name)
            sequence_time_path = os.path.join(base_dir, folder_name, sequence_time_name)    
            i = 10  
        else:
                 
            folder_name = f"sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok_record_only"
            sequence_length_name = f"sp_dy_gamma_etp_output_length_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
            sequence_time_name = f"sp_dy_gamma_etp_time_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
            sequence_length_path = os.path.join(base_dir, folder_name, sequence_length_name)
            sequence_time_path = os.path.join(base_dir, folder_name, sequence_time_name)    
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
                averages[i] = avg_value
        else:
            if os.path.exists(sequence_time_path) is False:
                print(f"No valid data in file: {sequence_time_path.split(f'/{dataset_name}/')[-1]}")
            elif os.path.exists(sequence_length_path) is False:
                print(f"No valid data in file: {sequence_length_path.split(f'/{dataset_name}/')[-1]}")

    # 儲存實驗結果
    speed_averages[experiment] = averages




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

print(f"speed_averages:{speed_averages}")
print(f"baseline_time_averages:{baseline_time_averages}")


plot_histogram(hist, bins,experiments,speed_averages,baseline_time_averages)





# for experiment in experiments:
#     averages = {}  # 儲存當前 experiment 的平均值
#     dynamic_averages = {}  # 儲存當前 experiment 在 dynamic gamma 下的平均值

#     for i in i_range:
#         # 構建標準實驗資料夾和檔案路徑
#         folder_name = f"sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok_record_only"
#         sequence_length_name = f"sp_dy_gamma_etp_output_length_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
#         sequence_time_name = f"sp_dy_gamma_etp_time_sp_dy_gamma_{i}_{experiment}_topkp0_fp16_2048tok.csv"
#         sequence_length_path = os.path.join(base_dir, folder_name, sequence_length_name)
#         sequence_time_path = os.path.join(base_dir, folder_name, sequence_time_name)    
#         # 檢查標準實驗檔案是否存在並計算平均值
#         if experiment not in ['70b_7b_68m','70b_7b_68m_adv'] :
#             if os.path.exists(sequence_time_path) and os.path.exists(sequence_length_path):
#                 with open(sequence_time_path, 'r') as f:
#                     data = f.readlines()
#                 time_values = [float(line.strip()) for line in data if line.strip()]  # 將資料轉換為浮點數列表
#                 with open(sequence_length_path, 'r') as f:
#                     data = f.readlines()
#                 length_values = [float(line.strip()) for line in data if line.strip()]  # 將資料轉換為浮點數列表
                
#                 if time_values or length_values:
#                     tot_time_array = np.array(length_values)/np.array(time_values) 
#                     avg_value = np.sum(np.array(length_values))/np.sum(tot_time_array)  # 計算平均值
#                     averages[i] = avg_value
#             else:
#                 if os.path.exists(sequence_time_path) is False:
#                     print(f"No valid data in file: {sequence_time_path.split(f'/{dataset_name}/')[-1]}")
#                 elif os.path.exists(sequence_length_path) is False:
#                     print(f"No valid data in file: {sequence_length_path.split(f'/{dataset_name}/')[-1]}")

#     # 儲存實驗結果
#     speed_averages[experiment] = averages



# baseline_time_averages = {}  # 儲存當前 experiment 的平均值
# for idx, experiment in enumerate(experiments_baseline):


#     folder_name = f"{experiment}_fp16_2048tok_record_only"
#     sequence_length_name = f"entropy_mesure_output_length_{experiment}_fp16_2048tok_record_only.csv"
#     sequence_time_name = f"entropy_mesure_time_{experiment}_fp16_2048tok_record_only.csv"
#     sequence_length_path = os.path.join(base_dir, folder_name, sequence_length_name)
#     sequence_time_path = os.path.join(base_dir, folder_name, sequence_time_name)    
    
#     # 檢查標準實驗檔案是否存在並計算平均值
#     if os.path.exists(sequence_time_path) and os.path.exists(sequence_length_path):
#         with open(sequence_time_path, 'r') as f:
#             data = f.readlines()
#         time_values = [float(line.strip()) for line in data if line.strip()]  # 將資料轉換為浮點數列表
#         with open(sequence_length_path, 'r') as f:
#             data = f.readlines()
#         length_values = [float(line.strip()) for line in data if line.strip()]  # 將資料轉換為浮點數列表
        
#         if time_values or length_values:
#             tot_time_array = np.array(length_values)/np.array(time_values) 
#             avg_value = np.sum(np.array(length_values))/np.sum(tot_time_array)  # 計算平均值
#         baseline_time_averages[experiment] = avg_value        
            
#     else:
#         if os.path.exists(sequence_time_path) is False:
#             print(f"No valid data in file: {sequence_time_path.split(f'/{dataset_name}/')[-1]}")
#         elif os.path.exists(sequence_length_path) is False:
#             print(f"No valid data in file: {sequence_length_path.split(f'/{dataset_name}/')[-1]}")           
#         baseline_time_averages[experiment] = None   
#         # 儲存實驗結果


# # 開始繪製圖表
# plt.figure(figsize=(10, 6))

# # 設置不同的顏色和標記
# colors = ['blue', 'green', 'red', 'black', 'black']
# markers = ['o', 's', '^', 'x', '*']

# # 繪製各個 experiment 的時間平均
# for idx, experiment in enumerate(experiments):
       
#     # if None in dynamic_time_averages[experiment].values():
#     #     continue
#     averages = speed_averages[experiment]
#     dynamic_averages = dynamic_time_averages[experiment]
    

#     x_values = sorted(averages.keys())  # i 值
#     y_values = [averages[i] for i in x_values]  # 標準實驗的平均值
#     dynamic_x_values = sorted(dynamic_averages.keys())  # i 值

#     if experiment not in ['70b_7b_68m','70b_7b_68m_adv'] :
#         dynamic_y_values = [dynamic_averages.get(i, np.nan) for i in x_values]  # dynamic gamma 的平均值
#     else:
#         dynamic_y_values = [dynamic_averages.get(i, np.nan) for i in dynamic_x_values]  # dynamic gamma 的平均值
#     if experiment not in ['70b_7b_68m','70b_7b_68m_adv'] :
#         plt.plot(x_values, y_values, label=f'SD ({experiment})', marker=markers[idx], color=colors[idx])
#         if np.isfinite(dynamic_y_values).all():
#             plt.plot(x_values, dynamic_y_values, label=f'Dynamic Gamma SD ({experiment})', marker=markers[idx], linestyle='--', color=colors[idx])
#     else:
#         plt.plot(dynamic_x_values, dynamic_y_values, label=f'Dynamic Gamma SD ({experiment})', marker=markers[idx], linestyle='--', color=colors[idx])        

#     if experiment not in  ['70b_7b_68m','70b_7b_68m_adv'] :
#         if np.isfinite(dynamic_y_values).all():
#             # 標記標準實驗的最大值
#             if len(y_values) > 0:
#                 max_y = max(y_values)
#                 max_x = x_values[np.argmax(y_values)]
#                 plt.text(max_x, max_y, f'{max_y:.2f}', ha='center', va='bottom', fontsize=10, color=colors[idx])
#                 # 標記 dynamic gamma 的最大值
#                 max_dynamic_y = max(dynamic_y_values)
#                 max_dynamic_x = x_values[np.argmax(dynamic_y_values)]
#                 plt.text(max_dynamic_x, max_dynamic_y, f'{max_dynamic_y:.2f}', ha='center', va='bottom', fontsize=10, color=colors[idx])
#     else:
#         # 標記 dynamic gamma 的最大值
#         if np.isfinite(dynamic_y_values).all():
#             max_dynamic_y = max(dynamic_y_values)
#             max_dynamic_x = dynamic_x_values[np.argmax(dynamic_y_values)]
#             plt.text(max_dynamic_x, max_dynamic_y, f'{max_dynamic_y:.2f}', ha='center', va='bottom', fontsize=10, color=colors[idx])
#         # pass        
    
# for idx, experiment in enumerate(experiments_baseline):
#     # if baseline_time_averages[experiment] is not None:
#     averages = baseline_time_averages[experiment]
#     y_values = [averages]*len(dynamic_x_values)
    
#     plt.plot(dynamic_x_values, y_values, label=f'Baseline Time Average ({experiment})', marker=markers[idx], linestyle='--', color='pink')



# # 添加標題和標籤
# plt.title('Speed Comparison and Dynamic Gamma')
# plt.xlabel('Gamma')
# plt.ylabel('Speed (toks/sec)')
# plt.legend()

# # 儲存圖像
# output_file = f"/work/valex1377/LLMSpeculativeSampling/{dataset_name}_Speed_dynamic_and_baseline_gamma_plot.png"
# plt.savefig(output_file)

# # 顯示圖像
# plt.show()

# print(f"Plot saved as {output_file}")





