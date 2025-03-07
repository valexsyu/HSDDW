import re

def count_numbers_and_words(file_path):
    total_numbers = 0
    total_words = 0

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # 匹配所有數字（包括整數和浮點數）
            numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', line)
            total_numbers += len(numbers)
            
            # 移除標籤如 [_START_xx_]，只計算純文字部分的單詞數
            clean_line = re.sub(r'\[_START_\d+_\]', '', line)
            words = re.findall(r'\b\w+\b', clean_line)
            total_words += len(words)

    return total_numbers, total_words



all_datasets=["mt_bench","humaneval","gsm8k","alpaca"]
for dataset in all_datasets:
    # 檔案路徑
    file_path_0 = f"/home/valexsyu/Documents/battleship/speculative_decoding/experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_temp1e-20_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_generated_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv"  # 替換為你的檔案名稱
    file_path_1 = f"/home/valexsyu/Documents/battleship/speculative_decoding/experiments/{dataset}/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_temp1_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_generated_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv"  # 替換為你的檔案名稱

    # 執行計算
    numbers_count_0, words_count_0 = count_numbers_and_words(file_path_0)

    # 輸出結果
    print(f"======{dataset}=======")
    print("Temp.=0")
    print(f"Total Numbers: {numbers_count_0}")
    print(f"Total Words: {words_count_0}")
    # 執行計算
    numbers_count, words_count = count_numbers_and_words(file_path_1)

    # 輸出結果
    print("Temp.=1")
    print(f"Total Numbers: {numbers_count}")
    print(f"Total Words: {words_count}")

    print(f"Ratio={(words_count_0-words_count)/words_count_0}")
