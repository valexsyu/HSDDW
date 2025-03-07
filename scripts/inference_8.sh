


# Function to set target and approx model based on the input model
set_model_names() {
    local model=$1
    if [[ "$model" == "70b_7b_68m" || "$model" == "70b_7b_68m_adv" || "$model" == "70b_68m" ]]; then
        target_model='llama-2-70b-chat'
        approx_model='llama-68m'
    elif [[ "$model" == "70b_7b" ]]; then
        target_model='llama-2-70b-chat'
        approx_model='llama-2-7b-chat'        
    elif [[ "$model" == "7b_68m" ]]; then
        target_model='llama-2-7b-chat'
        approx_model='llama-68m'
    else
        echo "Model name is wrong"
        exit 1
    fi
}




# #==========SD+dynamic gamma=============
# model='7b_68m'
# fn_name='sp_dy_gamma_etp'
# start_i=8
# end_i=13
# datasets_name=("humaneval" "gsm8k" "alpaca")
# # Call the function to set model names
# set_model_names $model

# for dataset_name in "humaneval" "gsm8k" "alpaca"
# do 
#     if [ "$dataset_name" != "gsm8k" ]; then
#         start_i=11
#     fi
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
# start_i=8
# end_i=13
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








# #==========SD=============
# model='7b_68m'
# fn_name='sp_dy_gamma_etp'
# start_i=14
# end_i=20
# # Call the function to set model names
# set_model_names $model

# for dataset_name in "alpaca"
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
# start_i=13
# end_i=13
# record_only=false
# start_num_data=0
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"



# # Example usage
# model='7b_1b'
# fn_name='sp_dy_gamma_etp'
# start_i=8
# end_i=13
# record_only=true
# start_num_data=54
# datasets_name=("gsm8k")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"



source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun3.sh
# # Example usage
# model='7b_1b'
# fn_name='sp_dy_gamma_etp'
# start_i=11
# end_i=13
# record_only=true
# start_num_data=119
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"

#==================================

# Example usage
model='7b_68m'
fn_name='sp_dy_gamma_etp'
start_i=8
end_i=9
record_only=false
start_num_data=0
datasets_name=("humaneval")
# Call the run_experiments function with the input parameters
run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"



model='7b_68m'
fn_name='sp_dy_gamma_etp'
start_i=10
end_i=10
record_only=false
start_num_data=0
datasets_name=("humaneval")
# Call the run_experiments function with the input parameters
run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"


model='7b_68m'
fn_name='sp_dy_gamma_etp'
start_i=14
end_i=14
record_only=false
start_num_data=0
datasets_name=("humaneval")
# Call the run_experiments function with the input parameters
run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"

