#!/bin/bash

# 定義目錄路徑
base_dir="/work/valex1377/LLMSpeculativeSampling/experiments/alpaca"
# 循環遍歷 i 從 2 到 20
for i in {2..20}
do
  # 定義原始資料夾名稱和新資料夾名稱
  old_folder="${base_dir}/sp_dy_gamma_${i}_7b_1b_topkp0_fp16_2048tok_record_only"
  new_folder="${base_dir}/sp_dy_gamma_${i}_7b_1b_topkp0_fp16_2048tok"
  
  # 檢查原始資料夾是否存在
  if [ -d "$old_folder" ]; then
    # 重命名資料夾
    mv "$old_folder" "$new_folder"
    echo "Renamed $old_folder to $new_folder"
  else
    echo "Folder $old_folder does not exist"
  fi
done




# #!/bin/bash

# # 接收目錄路徑作為參數
# folder_path=$1

# # 先處理目錄名稱中的 recode
# new_folder_path=$(echo "$folder_path" | sed 's/recode/record/g')

# # 如果新的目錄名稱和原來不同，則重命名目錄
# if [ "$folder_path" != "$new_folder_path" ]; then
#     mv "$folder_path" "$new_folder_path"
# fi

# # 遍歷新的目錄中的所有文件，將文件名中的 recode 替換為 record
# for file in "$new_folder_path"/*recode*; do
#     new_file_name=$(echo "$file" | sed 's/recode/record/g')
#     mv "$file" "$new_file_name"
# done
