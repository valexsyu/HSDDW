# #!/bin/bash

# # 定義目錄路徑
# base_dir="/work/valex1377/LLMSpeculativeSampling/experiments/mt_bench"

# # 循環遍歷 i 從 2 到 20
# for i in {2..20}
# do
#   # 定義原始資料夾名稱和新資料夾名稱
#   remove_folder="${base_dir}/sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok_recode_wrong"
  
#   # 檢查原始資料夾是否存在
#   if [ -d "$remove_folder" ]; then
#     # 重命名資料夾
#     rm -r "$remove_folder"
#     echo "Remove $remove_folder"
#   else
#     echo "Folder $remove_folder does not exist"
#   fi
# done
