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
#     --max_tokens 1 



# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     -b \
#     --max_tokens 1 \
#     --load_bits 16 \
#     --streaming_num 10 \
#     -m 5


# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     -b \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --streaming_num 30 \
#     --prefix_file_name  2to30 \
#     -m 5    


# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     -b \
#     --gamma 5 \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --streaming_num 30 \
#     --prefix_file_name  2to30_gamma5 \
#     -m 5    

# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     -b \
#     --gamma 6 \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --streaming_num 30 \
#     --prefix_file_name  2to30_gamma6 \
#     -m 5      

# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     -b \
#     --gamma 7 \
#     --max_tokens 1000 \
#     --load_bits 16 \
#     --streaming_num 30 \
#     --prefix_file_name  2to30_gamma7 \
#     -m 5         




# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b \
#     --gamma 5 \
#     --batch_mode 2 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --streaming_num 40 \
#     --prefix_file_name  2to40 \
#     -m 5  


# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
#     --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#     -b \
#     --gamma 5 \
#     --batch_mode 2 \
#     --max_tokens 500 \
#     --load_bits 16 \
#     --streaming_num 4 \
#     --prefix_file_name  testing \
#     -m 7



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
#     --streaming_num 5 \
#     --prefix_file_name  S5G4_uperbound_ \
#     -m 0    


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
#     --streaming_num 5 \
#     --prefix_file_name  test \
#     -m 8  
 




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
#     --prefix_file_name  68m_g4_200to200_1000GenWithEos_ \
#     -m 0



# for i in $(seq 10 10)
# do
#     #================2024/09/12 20:13 test sp_dy_gamma_etp_hrchl_adv
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok_tttttttttttt \
#         --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#         --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#         --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#         -r --record_time \
#         --entropy_th 0 \
#         --dataset mt_bench \
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

# for i in $(seq 10 10)
# do
#     #================2024/09/12 20:13 sp_dy_gamma_etp_hrchl_adv
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok_tttttttttttttttt \
#         --target_model_name llama-68m \
#         --approx_model_name llama-68m \
#         --dataset_name humaneval \
#         -r --record_time \
#         --entropy_th 0 \
#         --test_times 1 \
#         --gamma $i \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok \
#         -m 0  
# done


# for i in $(seq 11 12)
# do
#     #================2024/09/12 20:13 sp_dy_gamma_etp_hrchl_adv
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok \
#         --target_model_name llama-2-70b-chat \
#         --approx_model_name llama-68m \
#         --dataset_name mt_bench \
#         -r --record_time \
#         --entropy_th 0 \
#         --gamma $i \
#         --test_times 5 \
#         --model_70b_68m \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp_hrchl_adv \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_70b_7b_68m_adv_topkp0_fp16_2048tok \
#         -m 0  
# done




# echo "========================humaneval  70b_fp16_2048tok_recode_only===================="
# #================2024/08/29 13:53 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/humaneval/70b_fp16_2048tok_recode_only \
#     --target_model_name llama-2-70b-chat \
#     --approx_model_name llama-68m \
#     --model_70b_68m \
#     --dataset_name humaneval \
#     -r --record_time\
#     --gamma 1 \
#     --test_times 3 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name entropy_mesure \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  70b_fp16_2048tok_recode_only \
#     -m 0  




# echo "========================gsm8k      70b_fp16_2048tok_recode_only===================="
# #================2024/08/29 13:53 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/gsm8k/70b_fp16_2048tok_recode_only \
#     --target_model_name llama-2-70b-chat \
#     --approx_model_name llama-68m \
#     --model_70b_68m \
#     --dataset_name gsm8k \
#     -r --record_time\
#     --gamma 1 \
#     --test_times 3 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name entropy_mesure \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  70b_fp16_2048tok_recode_only \
#     --dataset_num_samples 150 \
#     -m 0  





# for i in $(seq 1 1)
# do
#     #================2024/07/17 20:08 mtbench
#     dataset_name="alpaca"
#     echo "==============sp_dy_gamma_${i}_70b_7b_topkp0_fp16_2048tok_recode_only==========="
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/${dataset_name}/70b_fp16_2048tok_recode_only \
#         --target_model_name llama-2-70b-chat \
#         --approx_model_name llama-68m \
#         --model_70b_68m \
#         --dataset_name ${dataset_name} \
#         --test_times 3 \
#         -r --record_time \
#         --entropy_th 0 \
#         --gamma $i \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name entropy_mesure \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  70b_fp16_2048tok_recode_only \
#         --dataset_num_samples 150 \
#         -m 0 
# done






#==================================

# Set different parameters

# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun6.sh
# model='68m_68m_68m'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=7
# end_i=7
# use_dy_gamma=true
# cal_entropy=true
# start_num_data=0
# datasets_name=("mt_bench")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"






# fn_name='at'
# start_num_data=0
# root_path='/work/valex1377/LLMSpeculativeSampling/experiments'
# target_model='llama-68m'
# topk=0
# p=0
# datasets_name=("mt_bench" "humaneval" "gsm8k" "alpaca")
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
# model='70b_7b_68m_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=10
# end_i=10
# use_dy_gamma=true
# cal_entropy=true
# temperature=1
# start_num_data=0
# datasets_name=("mt_bench_multi")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"


# # 20241126 mt_bench_multi for arr reveiwer2
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun8.sh
# model='70b_7b_68m_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=10
# end_i=10
# use_dy_gamma=true
# cal_entropy=true
# temperature=1e-20
# start_num_data=0
# datasets_name=("mt_bench_multi")
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
# datasets_name=("mt_bench")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"

# # 20241215 temperature=0 
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun8.sh
# model='70b_7b_68m_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=10
# end_i=10
# use_dy_gamma=true
# cal_entropy=true
# temperature=1e-20
# start_num_data=0
# datasets_name=("mt_bench_multi")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"




#source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun7.sh
#model='70b_7b_68m_adv'
#fn_name='sp_dy_gamma_etp_hrchl_adv'
#start_i=10
#end_i=10
#use_dy_gamma=true
#cal_entropy=true
#temperature=1
#start_num_data=0
#datasets_name=("mt_bench")
## Call the run_experiments function with the input parameters
#run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"

# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun8.sh
# model='70b_7b_68m_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=10
# end_i=10
# use_dy_gamma=true
# cal_entropy=true
# temperature=1
# start_num_data=0
# datasets_name=("mt_bench_multi")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"



# # 20241227 temperature=0 
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun9.sh
# model='70b_7b'
# fn_name='sp'
# start_i=10
# end_i=10
# use_dy_gamma=false
# cal_entropy=false
# temperature=1
# start_num_data=5
# datasets_name=("mt_bench")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"


# # 20241227 temperature=0 
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun10.sh
# model='70b_7b'
# fn_name='sp'
# start_i=10
# end_i=10
# use_dy_gamma=false
# cal_entropy=false
# temperature=1e-20
# start_num_data=16
# datasets_name=("mt_bench")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"


# # 20241227 temperature=0 
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun10.sh
# model='70b_7b'
# fn_name='sp'
# start_i=10
# end_i=10
# use_dy_gamma=false
# cal_entropy=false
# temperature=1e-20
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
temperature=1
start_num_data=0
datasets_name=("humaneval")
# Call the run_experiments function with the input parameters
run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"

