import os
import numpy as np
import matplotlib.pyplot as plt

# 定義資料夾路徑
dataset='mt_bench'
base_dir = "/work/valex1377/LLMSpeculativeSampling/experiments/"+dataset
# 定義範圍
gamma = 10
experiments = ["70b_7b","7b_68m"]

# 初始化字典來儲存所有實驗的平均值
accepted_averages_all = {}
reject_averages_all = {}

for experiment in experiments:
    accepted_averages = []
    reject_averages = []


    folder_name = f"sp_dy_gamma_{gamma}_{experiment}_topkp0_fp16_2048tok_record_only"
    accepted_file_name = f"sp_dy_gamma_etp_accepted_entropy_sp_dy_gamma_{gamma}_{experiment}_topkp0_fp16_2048tok.csv"
    accepted_file_path = os.path.join(base_dir, folder_name, accepted_file_name)

    # 構建 reject 檔案的路徑
    reject_file_name = f"sp_dy_gamma_etp_reject_entropy_sp_dy_gamma_{gamma}_{experiment}_topkp0_fp16_2048tok.csv"
    reject_file_path = os.path.join(base_dir, folder_name, reject_file_name)
    # 計算 accepted 檔案的平均值
    if os.path.exists(accepted_file_path):
        with open(accepted_file_path, 'r') as f:
            data = f.readlines()
        for i,line in enumerate(data):
            values = [float(x) for x in line.strip().split(',') if x]
            
            # 計算平均值
            avg_value = np.mean(values)
            accepted_averages.append(avg_value)

    else:
        print(f"Accepted file not found: {accepted_file_path}")

    # 計算 reject 檔案的平均值
    if os.path.exists(reject_file_path):
        with open(reject_file_path, 'r') as f:
            data = f.readlines()

        for i,line in enumerate(data):
            
            values = [float(x) for x in line.strip().split(',') if x]
            # 計算平均值
            avg_value = np.mean(values)
            reject_averages.append(avg_value)

    else:
        print(f"Reject file not found: {reject_file_path}")

    # 儲存每個實驗的平均值
    accepted_averages_all[experiment] = accepted_averages
    reject_averages_all[experiment] = reject_averages

# 開始繪製圖表
plt.figure(figsize=(10, 6))

# 設置不同的顏色和標籤
colors = ['blue', 'green', 'red']
markers = ['o', 'x', '^']

# 繪製 accepted 和 reject 兩條線
for idx, experiment in enumerate(experiments):
    x_values = list(range(len(accepted_averages)))

    # 繪製 accepted 平均值
    plt.plot(x_values, accepted_averages_all[experiment], label=f'Accepted-{experiment}', marker=markers[idx], color='blue')
    
    # 繪製 reject 平均值
    plt.plot(x_values, reject_averages_all[experiment], label=f'Reject-{experiment}', linestyle='--', marker=markers[idx], color='red')

# 添加標題和標籤
plt.title('Accepted vs Reject Entropy')
plt.xlabel('Data Index', fontsize=14)
plt.ylabel('Average Entropy', fontsize=14)
plt.legend()

# 儲存圖像
output_file = f"/work/valex1377/LLMSpeculativeSampling/scripts/hir_sd_w_dy_wdw/figs/Entropy_vs_Gamma_{dataset}.png"
plt.savefig(output_file)

# 顯示圖像
plt.show()

print(f"Plot saved as {output_file}")
