
# for i in $(seq 12 20)
# do
#     echo "===========================i:${i} ====================================="
#     #================2024/07/17 20:08 mtbench
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/sp_dy_gamma_${i}_70b_7b_68m_topkp0_fp16_2048tok \
#         --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat \
#         --approx_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m \
#         --dataset_path /work/valex1377/LLMSpeculativeSampling/datasets/mt_bench/question_single.json \
#         -r --record_time \
#         --entropy_th 3 \
#         --gamma $i \
#         --model_70b_68m \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name sp_dy_gamma_etp_hrchl \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  sp_dy_gamma_${i}_70b_7b_68m_topkp0_fp16_2048tok \
#         -m 0  
# done


# #================2024/08/29 13:53 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/gsm8k/7b_fp16_2048tok_recode_only \
#     --target_model_name llama-2-7b-chat \
#     --approx_model_name llama-68m \
#     --dataset_name gsm8k \
#     -r --record_time\
#     --gamma 1 \
#     --test_times 1 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name entropy_mesure \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  7b_fp16_2048tok_recode_only \
#     -m 0  


# echo "========================gsm8k  7b_fp16_2048tok_recode_only===================="
# #================2024/08/29 13:53 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/gsm8k/68m_fp16_2048tok_recode_only \
#     --target_model_name llama-68m  \
#     --approx_model_name llama-68m \
#     --dataset_name gsm8k \
#     -r --record_time\
#     --gamma 1 \
#     --test_times 1 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name entropy_mesure \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  7b_fp16_2048tok_recode_only \
#     -m 0  






# echo "========================humaneval  7b_fp16_2048tok_recode_only===================="
# #================2024/08/29 13:53 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/humaneval/7b_fp16_2048tok_recode_only \
#     --target_model_name llama-2-7b-chat \
#     --approx_model_name llama-68m \
#     --dataset_name humaneval \
#     -r --record_time\
#     --gamma 1 \
#     --test_times 3 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name entropy_mesure \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  7b_fp16_2048tok_recode_only \
#     -m 0  


# echo "========================humaneval  68m_fp16_2048tok_recode_only===================="
# #================2024/08/29 13:53 mtbench
# python main_modify.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --file_root /work/valex1377/LLMSpeculativeSampling/experiments/humaneval/68m_fp16_2048tok_recode_only \
#     --target_model_name llama-68m  \
#     --approx_model_name llama-68m \
#     --dataset_name humaneval \
#     -r --record_time\
#     --gamma 1 \
#     --test_times 3 \
#     --top_p 0 --top_k 0 \
#     -s 123 \
#     --fn_name entropy_mesure \
#     --max_tokens 2048 \
#     --load_bits 16 \
#     --prefix_file_name  68m_fp16_2048tok_recode_only \
#     -m 0  







# for i in $(seq 1 1)
# do
#     dataset_name="alpaca"
#     echo "==============sp_dy_gamma_7b_fp16_2048tok_recode_only==========="

#     #================2024/08/29 13:53 mtbench
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/${dataset_name}/7b_fp16_2048tok_recode_only \
#         --target_model_name llama-2-7b-chat \
#         --approx_model_name llama-68m \
#         --dataset_name ${dataset_name} \
#         -r --record_time\
#         --gamma 1 \
#         --test_times 3 \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name entropy_mesure \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  7b_fp16_2048tok_recode_only \
#         --dataset_num_samples 150 \
#         -m 0  

# done


# for i in $(seq 1 1)
# do
#     #================2024/07/17 20:08 mtbench
#     dataset_name="alpaca"
#     echo "==============sp_dy_gamma_68m_fp16_2048tok_recode_only==========="

#     #================2024/08/29 13:53 mtbench
#     python main_modify.py \
#         --input "The quick brown fox jumps over the lazy " \
#         --file_root /work/valex1377/LLMSpeculativeSampling/experiments/${dataset_name}/68m_fp16_2048tok_recode_only \
#         --target_model_name llama-68m  \
#         --approx_model_name llama-68m \
#         --dataset_name ${dataset_name} \
#         -r --record_time\
#         --gamma 1 \
#         --test_times 3 \
#         --top_p 0 --top_k 0 \
#         -s 123 \
#         --fn_name entropy_mesure \
#         --max_tokens 2048 \
#         --load_bits 16 \
#         --prefix_file_name  68m_fp16_2048tok_recode_only \
#         --dataset_num_samples 150 \
#         -m 0  

# done







# # Function to set target and approx model based on the input model
# set_model_names() {
#     local model=$1
#     if [[ "$model" == "70b_7b_68m" || "$model" == "70b_7b_68m_adv" || "$model" == "70b_68m" ]]; then
#         target_model='llama-2-70b-chat'
#         approx_model='llama-68m'
#     elif [[ "$model" == "70b_7b" ]]; then
#         target_model='llama-2-70b-chat'
#         approx_model='llama-2-7b-chat'        
#     elif [[ "$model" == "7b_68m" ]]; then
#         target_model='llama-2-7b-chat'
#         approx_model='llama-68m'
#     else
#         echo "Model name is wrong"
#         exit 1
#     fi
# }




# #==========SD+dynamic gamma=============
# model='7b_68m'
# fn_name='sp_dy_gamma_etp'
# start_i=2
# end_i=7
# datasets_name=("humaneval" "gsm8k" "alpaca")
# # Call the function to set model names
# set_model_names $model

# for dataset_name in "humaneval" "gsm8k" "alpaca"
# do 
#     for i in $(seq $start_i $end_i)
#     do
#         echo "=============================$dataset_name + $fn_name======================================"
#         echo "==============sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok==========="
        
#         # Set the dataset_num_samples option, only for non-humaneval datasets
#         dataset_num_samples_arg=""
#         if [ "$dataset_name" != "humaneval" ]; then
#             dataset_num_samples_arg="--dataset_num_samples 150"
#         fi
        
#         python main_modify.py \
#             --input "The quick brown fox jumps over the lazy " \
#             --file_root /work/valex1377/LLMSpeculativeSampling/experiments/${dataset_name}/sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok \
#             --target_model_name $target_model \
#             --approx_model_name $approx_model \
#             --dataset_name ${dataset_name} \
#             --test_times 3 \
#             -r --record_time \
#             --entropy_th 0 \
#             --gamma $i \
#             --top_p 0 --top_k 0 \
#             -s 123 \
#             --fn_name $fn_name \
#             --max_tokens 2048 \
#             --load_bits 16 \
#             --prefix_file_name sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok \
#             $dataset_num_samples_arg \
#             -m 0 
        
#         echo "=============================$dataset_name END======================================"
#         echo "==============sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok==========="                    
#     done
# done




# #==========SD=============
# model='7b_68m'
# fn_name='sp_dy_gamma_etp'
# start_i=2
# end_i=3
# datasets_name=("humaneval" "gsm8k" "alpaca")
# # Call the function to set model names
# set_model_names $model

# for dataset_name in "humaneval" "gsm8k" "alpaca"
# do 

#     for i in $(seq $start_i $end_i)
#     do
#         echo "=============================$dataset_name + $fn_name======================================"
#         echo "==============sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok_record_only==========="
        
#         # Set the dataset_num_samples option, only for non-humaneval datasets
#         dataset_num_samples_arg=""
#         if [ "$dataset_name" != "humaneval" ]; then
#             dataset_num_samples_arg="--dataset_num_samples 150"
#         fi
        
#         python main_modify.py \
#             --input "The quick brown fox jumps over the lazy " \
#             --file_root /work/valex1377/LLMSpeculativeSampling/experiments/${dataset_name}/sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok_record_only \
#             --target_model_name $target_model \
#             --approx_model_name $approx_model \
#             --dataset_name ${dataset_name} \
#             --test_times 3 \
#             -r --record_time \
#             --entropy_th 0 \
#             --gamma $i \
#             --top_p 0 --top_k 0 \
#             -s 123 \
#             --fn_name $fn_name \
#             --max_tokens 2048 \
#             --load_bits 16 \
#             --prefix_file_name sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok \
#             $dataset_num_samples_arg \
#             --record_only \
#             -m 0 
        
#         echo "=============================$dataset_name END======================================"
#         echo "==============sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok_record_only==========="                    
#     done
# done



# Source the common functions
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun2.sh


# # Example usage
# model='7b_1b'
# fn_name='sp_dy_gamma_etp'
# start_i=6
# end_i=7
# record_only=true
# start_num_data=129
# datasets_name=("gsm8k")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"

# # Example usage
# model='7b_1b'
# fn_name='sp_dy_gamma_etp'
# start_i=2
# end_i=7
# record_only=true
# start_num_data=0
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"

#==================================



# Source the common functions
source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun3.sh

# Example usage
model='1b'
fn_name='entropy_mesure'
start_i=1
end_i=1
record_only=true
start_num_data=0
datasets_name=("humaneval" "gsm8k" "alpaca")
# Call the run_experiments function with the input parameters
run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"



# Example usage
model='7b'
fn_name='entropy_mesure'
start_i=1
end_i=1
record_only=true
start_num_data=0
datasets_name=("humaneval" "gsm8k" "alpaca")
# Call the run_experiments function with the input parameters
run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"



