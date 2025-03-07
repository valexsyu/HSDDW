# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     -b \
#     --max_tokens 1 \


# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     -b \
#     --max_tokens 1 \
#     -m 3

# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     -b \
#     --gamma 4 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --streaming_num 2 \
#     --prefix_file_name  2 \
#     -m 5

# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     -b \
#     --gamma 4 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --streaming_num 2 \
#     --prefix_file_name  2\
#     -m 6



# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m  \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     -b \
#     --gamma 4 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  at_68m_ \
#     -m 2    





# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b \
#     --gamma 4 \
#     --batch_mode 2 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --streaming_num 2 \
#     --prefix_file_name  S2G4 \
#     -m 0    


# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b \
#     --gamma 7 \
#     --batch_mode 2 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --streaming_num 2 \
#     --prefix_file_name  S2G7 \
#     -m 0    



# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b \
#     --gamma 10 \
#     --batch_mode 2 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --streaming_num 2 \
#     --prefix_file_name  S2G10 \
#     -m 0    


# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     -b \
#     --gamma 4 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  SP_4_68m_10to3000_ \
#     -m 3






# echo "===============llama at=================="
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b \
#     -s 123 \
#     --batch_mode 0 \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  llama_1000GenWithEos \
#     -m 0    

# echo "===============llama-68m at=================="
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b \
#     -s 123 \
#     --batch_mode 0 \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  llama_68m_1000GenWithEos \
#     -m 0  




# ## 50 to 100
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
#     --prefix_file_name  68m_g4_200to200_1000GenWithEos_1_ \
#     -m 0



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
#     --prefix_file_name  68m_g4_180and180_1000GenWithEos_MTbench \
#     -m 0



# #sp_llama_68
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b -r \
#     --gamma 4 \
#     -s 123 \
#     --batch_mode 1 \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  68m_g4_1000GenWithEos_new \
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
#     --prefix_file_name  68m_g4_180and180_1000GenWithEos_MTbench \
#     -m 0




# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     -s 123 \
#     --fn_name sp_dy_gamma \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  68m \
#     -m 0  

# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_dy_68m_topkp0_KLlambda02 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     -s 123 \
#     --fn_name sp_dy_gamma \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  dy_68m_topkp0_KLlambda02 \
#     -m 0  

# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_dy_68m_topkp0_preGamma3 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     -s 123 \
#     --fn_name sp_dy_gamma \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  dy_68m_topkp0_preGamma3 \
#     -m 0  




# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_dyetp_th4 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_entropy_dyetp_th4 \
#     -m 0  


# ================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_68m_topkp0 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     -s 123 \
#     --fn_name spsp \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  sp_68m_topkp0 \
#     -m 0  


# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_68m_topkp0 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  sp_68m_topkp0_ \
#     -m 0  


# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_68m_topkp0_g10 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 10 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  sp_68m_topkp0_g10 \
#     -m 0  


# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_68m_topkp0_etp_golden_g4 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -gasp /work/valex1377/LLMSpeculativeSampling/experiments/sp_68m_topkp0/sp_accepted_sequence_sp_68m_topkp0_.csv \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  sp_68m_topkp0_etp_golden_g4 \
#     -m 0  




# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_68m_topkp0_fp32 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  sp_68m_topkp0__fp32 \
#     -m 0  

# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_68m_topkp0_fp32_500tok \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_68m_topkp0_fp32_500tok \
#     -m 0  


# ## f32, n position and every step, deatch,
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_fp32_500tok_nPos \
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
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_fp32_500tok_nPos \
#     -m 0   



# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70_7b_topkp0_fp16_500tok_entory_record_only \
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
#     --prefix_file_name  sp_dy_gamma_70_7b_topkp0_fp16_500tok_entory_record_only \
#     -m 0  


# #================2024/05/31 12:xx mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70_68m_topkp0_fp16_500tok_entory_recordonly \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_70_68m_topkp0_fp16_500tok_entory_recordonly \
#     -m 0  




# #================2024/05/31 16:11 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70_7b_topkp0_fp16_500tok_entory_recordonly \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_70_7b_topkp0_fp16_500tok_entory_recordonly \
#     -m 0  


# #================2024/07/17 19:35 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_fp32_500tok_entropy_dyetp_th4_new \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 32 \
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_fp32_500tok_entropy_dyetp_th4 \
#     -m 0  

# #================2024/07/17 19:35 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_68m_topkp0_fp32_500tok_new \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time\
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp \
#     --max_tokens 500 \
#     --load_bits 32 \
#     --prefix_file_name  sp_68m_topkp0_fp32_500tok \
#     -m 0  


# #================2024/07/17 19:35 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_fp32_500tok_entropy_dyetp_th3_new \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 32 \
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_fp32_500tok_entropy_dyetp_th3 \
#     -m 0  




# #================2024/07/17 19:35 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_70b_68m_topkp0_fp16_500tok \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time\
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_70b_68m_topkp0_fp16_500tok \
#     -m 0  




# #================2024/07/17 19:35 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_70b_7b_topkp0_fp16_500tok \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time\
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_70b_7b_topkp0_fp16_500tok \
#     -m 0  



# #================2024/07/17 19:35 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70b_68m_topkp0_fp32_500tok_entropy_dyetp_th35 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3.5 \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_70b_68m_topkp0_fp32_500tok_entropy_dyetp_th35 \
#     -m 0 



# #================2024/07/17 19:35 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70b_68m_topkp0_fp32_500tok_entropy_dyetp_th42 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 4.2 \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_70b_68m_topkp0_fp32_500tok_entropy_dyetp_th42 \
#     -m 0 






# #================2024/08/29 13:53 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/70b_fp16_4096tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time\
#     --model_70b_68m \
#     --gamma 1 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name entropy_mesure \
#     --max_tokens 4096 \
#     --load_bits 16 \
#     --prefix_file_name  70b_fp16_4096tok_recode_only \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_5_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 5 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_5_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_3_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 3 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_3_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_2_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 2 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_2_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_4_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_4_70b_68m_topkp0_fp16_2048tok \
#     -m 0  

# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_6_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 6 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_6_70b_68m_topkp0_fp16_2048tok \
#     -m 0  



# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_11_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 11 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_11_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_12_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 12 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_12_70b_68m_topkp0_fp16_2048tok \
#     -m 0  






# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_13_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 13 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_13_70b_68m_topkp0_fp16_2048tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_14_70b_68m_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --model_70b_68m \
#     --entropy_th 3 \
#     --gamma 14 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_14_70b_68m_topkp0_fp16_2048tok \
#     -m 0  



# for i in $(seq 2 10)
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


# for i in $(seq 6 9)
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

# for i in $(seq 6 9)
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





# for i in $(seq 6 9)
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
#         -m 0 
# done




# for i in $(seq 6 9)
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


# for i in $(seq 8 9)
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




# for i in $(seq 6 9)
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

# for i in $(seq 6 9)
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





# Source the common functions
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun.sh


# #==================================
# # Example usage
# model='70b_7b'
# fn_name='sp_dy_gamma_etp'
# start_i=9
# end_i=9
# datasets_name=("gsm8k")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i "${datasets_name[@]}"

# #==================================
# # Example usage
# model='70b_7b'
# fn_name='sp_dy_gamma_etp'
# start_i=6
# end_i=9
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i "${datasets_name[@]}"




#==================================
# Source the common functions
source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun2.sh

# # Set different parameters
# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp_hrchl'
# start_i=7
# end_i=9
# record_only=false
# start_num_data=10
# datasets_name=("gsm8k")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"


# # Set different parameters
# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp_hrchl'
# start_i=9
# end_i=9
# record_only=false
# start_num_data=123
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"


# # Example usage
# model='70b_7b_1b_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=6
# end_i=9
# record_only=false
# start_num_data=0
# datasets_name=("humaneval" "gsm8k" "alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"



# # Source the common functions
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun4.sh

# # Example usage
# model='70b_7b'
# fn_name='sp'
# start_i=10
# end_i=10
# use_dy_gamma=false
# start_num_data=0
# datasets_name=("mt_bench" "humaneval" "gsm8k" "alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $start_num_data "${datasets_name[@]}"


# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun5.sh
# model='70b_7b'
# fn_name='sp'
# start_i=10
# end_i=10
# use_dy_gamma=false
# cal_entropy=false
# start_num_data=0
# datasets_name=("mt_bench" "humaneval" "gsm8k" "alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"


# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun2.sh
# # Example usage
# model='70b_7b_1b_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=7
# end_i=9
# record_only=false
# start_num_data=0
# datasets_name=("alpaca")
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
# datasets_name=("humaneval")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"




# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun6.sh
# model='70b_7b'
# fn_name='sp'
# start_i=10
# end_i=10
# use_dy_gamma=false
# cal_entropy=false
# start_num_data=0
# datasets_name=("mt_bench" "humaneval")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"



# # 20241126 temperature for arr reveiwer3
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun8.sh
# model='70b'
# fn_name='at'
# start_i=10
# end_i=10
# use_dy_gamma=false
# cal_entropy=false
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
# start_num_data=147
# datasets_name=("gsm8k")
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
# temperature=1
# start_num_data=0
# datasets_name=("gsm8k")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"


# # 20241227 temperature=1
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun10.sh
# model='70b_7b'
# fn_name='sp'
# start_i=10
# end_i=10
# use_dy_gamma=false
# cal_entropy=false
# temperature=1
# start_num_data=0
# datasets_name=("gsm8k")
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
datasets_name=("humaneval")
# Call the run_experiments function with the input parameters
run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"



