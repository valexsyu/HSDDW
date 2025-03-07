import csv

def process_csv_files(length_file, accepted_file):
    def read_csv_and_sum(file_path):
        total = 0
        with open(file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # 將每行的所有數字加總
                total += sum(map(int, filter(None, row)))
        return total

    # 計算總長度和總接受數量
    total_length = read_csv_and_sum(length_file)
    total_accepted = read_csv_and_sum(accepted_file)

    # 計算比值
    if total_accepted == 0:
        ratio = float("inf")  # 避免除以零
    else:
        ratio = total_accepted/total_length

    return total_length, total_accepted, ratio

# 檔案路徑
# length_file_path = "/home/valexsyu/Documents/battleship/speculative_decoding/experiments/gsm8k/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_temp1_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_gamma_sequence_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv"  # 替換為你的長度檔案名稱
# accepted_file_path = "/home/valexsyu/Documents/battleship/speculative_decoding/experiments/gsm8k/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_temp1_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_accepted_sequence_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv"  # 替換為你的接受數檔案名稱

length_file_path = "/home/valexsyu/Documents/battleship/speculative_decoding/experiments/gsm8k/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_temp1e-20_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_gamma_sequence_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv"  # 替換為你的長度檔案名稱
accepted_file_path = "/home/valexsyu/Documents/battleship/speculative_decoding/experiments/gsm8k/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_temp1e-20_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_accepted_sequence_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv"  # 替換為你的接受數檔案名稱


# 執行處理
total_length, total_accepted, length_to_accepted_ratio = process_csv_files(length_file_path, accepted_file_path)

# 顯示結果
print(f"Total Length: {total_length}")
print(f"Total Accepted Number: {total_accepted}")
print(f"Length / Accepted Number: {length_to_accepted_ratio:.4f}")
