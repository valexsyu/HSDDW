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
#     --streaming_num 5 \
#     -m 4



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
#     --streaming_num 2 \
#     --prefix_file_name  testing \
#     -m 7  





# ## 150 to 200
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
#     --prefix_file_name  68m_g4_200to200_1000GenWithEos_3_ \
#     -m 0

# ##local input #================2024/04/17 23:34 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b -r \
#     -s 123 \
#     --gamma 4 \
#     --batch_mode 4 \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --prefix_file_name  68m_g4_400_1000GenWithEos_MtBench \
#     -m 0

    


# ## f32, all position and every step, deatch, lr_schedule
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_lrSch \
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
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_fp32_500tok_allPos_lrSch \
#     -m 0   


# ## n position and each 20 step get the loss + update , deatch
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_68m_topkp0_fp32_500tok_nPos_s20Loss20Ud \
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
#     --prefix_file_name  sp_dy_gamma_68m_topkp0_fp32_500tok_nPos_s20Loss20Ud \
#     -m 0   








# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_68m_topkp0_fp32_500tok_golden_g4 \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -gasp /work/valex1377/LLMSpeculativeSampling/experiments/sp_68m_topkp0_fp32_500tok/sp_accepted_sequence_sp_68m_topkp0_fp32_500tok.csv \
#     -r --record_time \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --prefix_file_name  sp_68m_topkp0_fp32_500tok_golden_g4 \
#     -m 0  





# #================2024/04/22 11:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70_68m_topkp0_fp16_500tok_entory_dyetp_th4 \
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
#     --prefix_file_name  sp_dy_gamma_70_68m_topkp0_fp16_500tok_entory_dyetp_th4 \
#     -m 0  


# #================2024/05/31 11:01 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_70_68m_topkp0_fp16_500tok_entory_fix_th4 \
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
#     --prefix_file_name  sp_dy_gamma_70_68m_topkp0_fp16_500tok_entory_fix_th4 \
#     -m 0  





# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_5_70b_7b_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --entropy_th 3 \
#     --gamma 5 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_5_70b_7b_topkp0_fp16_2048tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_3_70b_7b_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --entropy_th 3 \
#     --gamma 3 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_3_70b_7b_topkp0_fp16_2048tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_2_70b_7b_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --entropy_th 3 \
#     --gamma 2 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_2_70b_7b_topkp0_fp16_2048tok \
#     -m 0  


# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_4_70b_7b_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --entropy_th 3 \
#     --gamma 4 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_4_70b_7b_topkp0_fp16_2048tok \
#     -m 0  

# #================2024/07/17 20:08 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_6_70b_7b_topkp0_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time \
#     --entropy_th 3 \
#     --gamma 6 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name sp_dy_gamma_etp \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  sp_dy_gamma_6_70b_7b_topkp0_fp16_2048tok \
#     -m 0  



# for i in $(seq 2 10)
# do
#     #================2024/07/17 20:08 mtbench
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok \
#         --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#         --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#         --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
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




# for i in $(seq 2 10)
# do
#     #================2024/07/17 20:08 mtbench
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_${i}_7b_68m_topkp0_fp16_2048tok \
#         --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#         --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#         --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#         -r --record_time \
#         --entropy_th 0 \
#         --gamma $i \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_7b_68m_topkp0_fp16_2048tok \
#         -m 0 
# done



# for i in $(seq 11 20)
# do
#     #================2024/07/17 20:08 mtbench
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_${i}_7b_68m_topkp0_fp16_2048tok_qq \
#         --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#         --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#         --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#         -r --record_time \
#         --entropy_th 0 \
#         --gamma $i \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_7b_68m_topkp0_fp16_2048tok \
#         -m 0 
# done




# for i in $(seq 2 11)
# do
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok_recode_only \
#         --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#         --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#         --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#         -r --record_time \
#         --entropy_th 3 \
#         --gamma ${i} \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok \
#         -m 0  

# done



# #================2024/08/29 13:53 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/70b_fp16_2048tok_recode_only \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -r --record_time\
#     --model_70b_68m \
#     --gamma 1 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name entropy_mesure \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  70b_fp16_2048tok_recode_only \
#     -m 0  



# for i in $(seq 14 17)
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


# for i in $(seq 14 17)
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






# for i in $(seq 14 17)
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




# for i in $(seq 14 17)
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


# for i in $(seq 14 17)
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



# for i in $(seq 14 17)
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

# for i in $(seq 14 17)
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
# start_i=14
# end_i=17
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i "${datasets_name[@]}"




# #==================================

# # Set different parameters
# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp_hrchl'
# start_i=14
# end_i=17
# datasets_name=("humaneval" "gsm8k" "alpaca")

# # Call the run_experiments function with different parameters
# run_experiments $model $fn_name $start_i $end_i "${datasets_name[@]}"


# # Source the common functions
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun3.sh
# # Example usage
# model='70b_7b_1b_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=14
# end_i=17
# record_only=false
# start_num_data=0
# datasets_name=("humaneval" "gsm8k" "alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"




# # Source the common functions
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun4.sh

# # Example usage
# model='70b_1b'
# fn_name='sp'
# start_i=10
# end_i=10
# use_dy_gamma=false
# start_num_data=0
# datasets_name=("mt_bench" "humaneval" "gsm8k" "alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $start_num_data "${datasets_name[@]}"



# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun5.sh
# model='70b_1b'
# fn_name='sp'
# start_i=10
# end_i=10
# use_dy_gamma=false
# cal_entropy=false
# start_num_data=0
# datasets_name=("mt_bench" "humaneval" "gsm8k" "alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"


# # Source the common functions
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun3.sh
# # Example usage
# model='70b_7b_1b_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=16
# end_i=17
# record_only=false
# start_num_data=0
# datasets_name=("gsm8k")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"


# # Example usage
# model='70b_7b_1b_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=14
# end_i=17
# record_only=false
# start_num_data=0
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"


# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun6.sh
# model='70b'
# fn_name='sp'
# start_i=1
# end_i=1
# use_dy_gamma=false
# cal_entropy=true
# start_num_data=0
# datasets_name=("mt_bench")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"




# fn_name='at'
# start_num_data=0
# root_path='/work/valex1377/LLMSpeculativeSampling/experiments'
# target_model='llama-2-70b-chat'
# topk=0
# p=0
# datasets_name=("mt_bench")
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


# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun6.sh
# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp_hrchl'
# start_i=10
# end_i=10
# use_dy_gamma=false
# cal_entropy=false
# start_num_data=0
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"








# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun6.sh
# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp_hrchl'
# start_i=10
# end_i=10
# use_dy_gamma=false
# cal_entropy=false
# start_num_data=0
# datasets_name=("gsm8k")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"



# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun6.sh
# model='70b_7b_68m_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=10
# end_i=10
# use_dy_gamma=true
# cal_entropy=true
# start_num_data=0
# datasets_name=("gsm8k")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"







