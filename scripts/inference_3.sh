# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     -b \
#     --max_tokens 1 \


# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     -b \
#     --max_tokens 1 \
#     --load_bits 16 \
#     --streaming_num 10 \
#     -m 4

# echo "===============inference_3.sh=================="
# echo "===============sp68m=================="

# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b \
#     --gamma 4 \
#     --batch_mode 1 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  68m \
#     -m 0       


# echo "===============llama at=================="
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b \
#     --gamma 4 \
#     --batch_mode 0 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  llama \
#     -m 0    

# echo "===============llama-68m at=================="
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b \
#     --gamma 4 \
#     --batch_mode 0 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  llama_68m \
#     -m 0   


# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     -b \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  at_68m_10to3000_ \
#     -m 2   
    
     


# ## 100 to 150
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b -r \
#     -s 123 \
#     --gamma 4 \
#     --batch_mode 3 \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  68m_g4_200to200_1000GenWithEos_2_ \
#     -m 0     


# #================2024/04/17 23:34 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r \
#     -s 123 \
#     --gamma 4 \
#     --batch_mode 3 \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  68m_g4_0and400_1000GenWithEos_MTbench \
#     -m 0



# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_dy_68m_topkp0_KLlambda01 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     -s 123 \
#     --fn_name sp_dy_gamma \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  dy_68m_topkp0_KLlambda01 \
#     -m 0  


# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_dy_68m_topkp0_M3P3 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     -s 123 \
#     --fn_name sp_dy_gamma \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  dy_68m_topkp0_M3P3 \
#     -m 0  



# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_dyetp_th5 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_entropy_dyetp_th5 \
#     -m 0  



# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_dyetp_th4_g15 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 15 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_entropy_dyetp_th4_g15 \
#     -m 0  


# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/test \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --use_apt_param \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp_grad \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  test \
#     -m 0  

# ## f32, all position and every step, deatch,
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_fp32_500tok_allPos \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --use_apt_param \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp_grad \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_fp32_500tok_allPos \
#     -m 0   


# ## n position and each loss + 20 update , deatch
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_fp32_500tok_nPos_s1Loss20Ud \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --use_apt_param \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp_grad \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_fp32_500tok_nPos_s1Loss20Ud \
#     -m 0   


# ## n position and each 20 step get the loss + update , deatch
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_s20Loss20Ud \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --use_apt_param \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp_grad \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_s20Loss20Ud \
#     -m 0   


# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_fp32_500tok_entropy_dyetp_th4 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_fp32_500tok_entropy_dyetp_th4 \
#     -m 0  


# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70_68m_topkp0_fp32_entory_record_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_70_68m_topkp0_fp32_entory_record_only \
#     -m 0  



# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70_68m_topkp0_fp32_entory_dyetp_th4 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_70_68m_topkp0_fp32_entory_dyetp_th4 \
#     -m 0  


# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70_7b_topkp0_fp16_500tok_entory_dyetp_th07 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_70_7b_topkp0_fp16_500tok_entory_dyetp_th07 \
#     -m 0  






# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_fp32_500tok_entropy_dyetp_th4_new \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 32 \
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_fp32_500tok_entropy_dyetp_th4 \
#     -m 0  



# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70b_7b_topkp0_fp16_500tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat  \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 32 \
#     --prefix_file_name  sp_dy_gamma_70b_7b_topkp0_fp16_500tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70b_7b_topkp0_fp16_500tok_entropy_dyetp_th1 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat  \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --entropy_th 1 \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_70b_7b_topkp0_fp16_500tok_entropy_dyetp_th1 \
#     -m 0  



# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70b_7b_topkp0_fp16_500tok_entropy_dyetp_th11 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat  \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --entropy_th 1.1 \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_70b_7b_topkp0_fp16_500tok_entropy_dyetp_th11 \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70b_7b_topkp0_fp16_500tok_entropy_dyetp_th09 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat  \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --entropy_th 0.9 \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_70b_7b_topkp0_fp16_500tok_entropy_dyetp_th09 \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70b_7b_topkp0_fp16_500tok_entropy_dyetp_th08 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat  \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --entropy_th 0.8 \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_70b_7b_topkp0_fp16_500tok_entropy_dyetp_th08 \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70b_7b_topkp0_fp16_500tok_entropy_dyetp_th12 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat  \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --entropy_th 1.2 \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_70b_7b_topkp0_fp16_500tok_entropy_dyetp_th12 \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_7_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 7 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_7_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


#     #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_8_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 8 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_8_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_9_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 9 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_9_70b_68m_topkp0_fp16_2048tok \
#     -m 0  

# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 10 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_10_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_15_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 15 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_15_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_16_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 16 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_16_70b_68m_topkp0_fp16_2048tok \
#     -m 0  






# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_17_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 17 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_17_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_18_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 18 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_18_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


#     #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_19_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 19 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_19_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


#     #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_20_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 20 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_20_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


# for i in $(seq 11 20)
# do
#     #================2024/07/17 20:08 mtbench
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_${i}_70b_68m_topkp0_fp16_2048tok_recode_only \
#         --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#         --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#         --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#         -r --record_time \
#         --model_70b_68m \
#         --entropy_th 0 \
#         --gamma $i \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_70b_68m_topkp0_fp16_2048tok \
#         -m 0  
# done    


# for i in $(seq 10 13)
# do
#     #================2024/09/12 20:13 sp_dy_gamma_etp_hrchl_adv
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok \
#         --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#         --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#         --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#         -r --record_time \
#         --entropy_th 0 \
#         --gamma $i \
#         --model_70b_68m \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp_hrchl_adv \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok \
#         -m 0  
# done

# for i in $(seq 10 13)
# do
#     #================2024/09/12 20:13 sp_dy_gamma_etp_hrchl_adv
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/humaneval/sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok \
#         --target_model_name llama-2-70b-chat \
#         --approx_model_name llama-68m \
#         --dataset_name humaneval \
#         -r --record_time \
#         --entropy_th 0 \
#         --gamma $i \
#         --test_times 3 \
#         --model_70b_68m \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp_hrchl_adv \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok \
#         -m 0  
# done





# for i in $(seq 10 13)
# do
#     dataset_name="humaneval"
#     #================2024/07/17 20:08 mtbench
#     echo "==============sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok_recode_only==========="
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/${dataset_name}/sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok_recode_only \
#         --target_model_name llama-2-70b-chat \
#         --approx_model_name llama-2-7b-chat \
#         --dataset_name ${dataset_name} \
#         --test_times 3 \
#         -r --record_time \
#         --entropy_th 0 \
#         --gamma $i \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok \
#         --record_only \
#         -m 0 
# done



# for i in $(seq 10 13)
# do
#     #================2024/09/12 20:13 sp_dy_gamma_etp_hrchl_adv
#     echo "========================sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok===================="
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/gsm8k/sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok \
#         --target_model_name llama-2-70b-chat \
#         --approx_model_name llama-68m \
#         --dataset_name gsm8k \
#         -r --record_time \
#         --entropy_th 0 \
#         --gamma $i \
#         --test_times 3 \
#         --model_70b_68m \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp_hrchl_adv \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok \
#         --dataset_num_samples 150 \
#         -m 0  
# done



# for i in $(seq 10 13)
# do
#     #================2024/07/17 20:08 mtbench
#     dataset_name="gsm8k"
#     echo "==============sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok_recode_only==========="
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/${dataset_name}/sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok_recode_only \
#         --target_model_name llama-2-70b-chat \
#         --approx_model_name llama-2-7b-chat \
#         --dataset_name ${dataset_name} \
#         --test_times 3 \
#         -r --record_time \
#         --entropy_th 0 \
#         --gamma $i \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok \
#         --dataset_num_samples 150 \
#         --record_only \
#         -m 0 
# done



# for i in $(seq 10 13)
# do
#     #================2024/09/12 20:13 sp_dy_gamma_etp_hrchl_adv
#     dataset_name="alpaca"
#     echo "========================sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok===================="
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/${dataset_name}/sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok \
#         --target_model_name llama-2-70b-chat \
#         --approx_model_name llama-68m \
#         --dataset_name ${dataset_name} \
#         -r --record_time \
#         --entropy_th 0 \
#         --gamma $i \
#         --test_times 3 \
#         --model_70b_68m \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp_hrchl_adv \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok \
#         --dataset_num_samples 150 \
#         -m 0  
# done

# for i in $(seq 10 13)
# do
#     #================2024/07/17 20:08 mtbench
#     dataset_name="alpaca"
#     echo "==============sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok_recode_only==========="
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/${dataset_name}/sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok_recode_only \
#         --target_model_name llama-2-70b-chat \
#         --approx_model_name llama-2-7b-chat \
#         --dataset_name ${dataset_name} \
#         --test_times 3 \
#         -r --record_time \
#         --entropy_th 0 \
#         --gamma $i \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok \
#         --dataset_num_samples 150 \
#         --record_only \
#         -m 0 
# done







# # Source the common functions
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun.sh



# #==================================
# # Example usage
# model='70b_7b'
# fn_name='sp_dy_gamma_etp'
# start_i=13
# end_i=13
# datasets_name=("humaneval")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i "${datasets_name[@]}"


# #==================================
# # Example usage
# model='70b_7b'
# fn_name='sp_dy_gamma_etp'
# start_i=11
# end_i=13
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i "${datasets_name[@]}"



# Source the common functions
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun2.sh
#==================================

# # Set different parameters
# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp_hrchl'
# start_i=12
# end_i=13
# record_only=false
# start_num_data=115
# datasets_name=("humaneval")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"

# # Set different parameters
# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp_hrchl'
# start_i=10
# end_i=13
# record_only=false
# start_num_data=0
# datasets_name=("gsm8k" "alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"


# # Source the common functions
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun3.sh
# # Example usage
# model='70b_7b_1b_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=10
# end_i=13
# record_only=false
# start_num_data=160
# datasets_name=("humaneval")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"

# # Example usage
# model='70b_7b_1b_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=10
# end_i=13
# record_only=false
# start_num_data=0
# datasets_name=("gsm8k" "alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"



# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun6.sh
# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp_hrchl'
# start_i=5
# end_i=5
# use_dy_gamma=false
# cal_entropy=true
# start_num_data=0
# datasets_name=("gsm8k")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"






# fn_name='at'
# start_num_data=0
# root_path='/work/valex1377/LLMSpeculativeSampling/experiments'
# target_model='llama-2-70b-chat'
# topk=0
# p=0
# datasets_name=("gsm8k")
# # Call the run_experiments function with the input parameters
# for dataset_name in "${datasets_name[@]}"
# do 
#     # Call the function to set model names
#     dataset_num_samples_arg=""
#     if [[ "$dataset_name" == "gsm8k" || "$dataset_name" == "alpaca" ]]; then
#         dataset_num_samples_arg="--dataset_num_samples 150"
#     fi


#     echo "=========$dataset_name + $fn_name Start======${target_model}_topk${topk}p${p}_fp16_2048tok==========="
    
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root $root_path/${dataset_name}/${target_model}_topk${topk}p${p}_fp16_2048tok \
#         --target_model_name $target_model \
#         --dataset_name ${dataset_name} \
#         --test_times 3 \
#         -r --record_time \
#         --entropy_th 0 \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name $fn_name \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name "" \
#         $dataset_num_samples_arg \
#         --start_num_data $start_num_data \
#         -m 0 
    
#     echo "=========$dataset_name END  =====${i}_${model}_topkp20_fp16_2048tok${use_dy_gamma_suffix}==========="                    

# done



# # 20241126 mt_bench_multi for arr reveiwer2
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun8.sh
# model='70b_7b'
# fn_name='sp'
# start_i=10
# end_i=10
# use_dy_gamma=false
# cal_entropy=true
# temperature=1
# start_num_data=0
# datasets_name=("mt_bench_multi")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"


# # 20241213 temperature=0 
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun7.sh
# model='70b_7b_68m_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=10
# end_i=10
# use_dy_gamma=true
# cal_entropy=true
# temperature=1e-20
# start_num_data=0
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"


# # 20241215 temperature=0 
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun7.sh
# model='70b_7b_68m_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=10
# end_i=10
# use_dy_gamma=true
# cal_entropy=true
# temperature=1e-20
# start_num_data=0
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"









# # 20241218 temperature=1
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun7.sh
# model='70b_7b_68m_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=10
# end_i=10
# use_dy_gamma=true
# cal_entropy=true
# temperature=1
# start_num_data=0
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"


# # 20241215 temperature=0 
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun7.sh
# model='68m_68m_68m'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=10
# end_i=10
# use_dy_gamma=true
# cal_entropy=true
# temperature=1e-20
# start_num_data=0
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"


# 20241227 temperature=0 
source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun10.sh
model='70b_7b'
fn_name='sp'
start_i=10
end_i=10
use_dy_gamma=false
cal_entropy=false
temperature=1e-20
start_num_data=0
datasets_name=("alpaca")
# Call the run_experiments function with the input parameters
run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"
