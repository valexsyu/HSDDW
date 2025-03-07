
# for i in $(seq 2 11)
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




# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun3.sh


# # Set different parameters
# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp_hrchl'
# start_i=10
# end_i=10
# record_only=true
# start_num_data=0
# datasets_name=("humaneval")
# # Call the run_experiments function with different parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"

# # Set different parameters
# model='70b_7b_68m_adv'
# fn_name='sp_dy_gamma_etp_hrchl_adv'
# start_i=10
# end_i=10
# record_only=true
# start_num_data=0
# datasets_name=("humaneval")
# # Call the run_experiments function with different parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"




# # Set different parameters
# model='70b_68m'
# fn_name='sp_dy_gamma_etp'
# start_i=10
# end_i=10
# record_only=true
# start_num_data=0
# datasets_name=("humaneval")
# # Call the run_experiments function with different parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"







# # Source the common functions
# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun5.sh
# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp_hrchl'
# start_i=5
# end_i=5
# use_dy_gamma=false
# cal_entropy=false
# start_num_data=0
# datasets_name=("mt_bench" "humaneval" "gsm8k" "alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"



# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp'
# start_i=3
# end_i=5
# record_only=false
# start_num_data=0
# datasets_name=("gsm8k")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $record_only $start_num_data "${datasets_name[@]}"





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





# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun6.sh
# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp_hrchl'
# start_i=5
# end_i=5
# use_dy_gamma=false
# cal_entropy=false
# start_num_data=0
# datasets_name=("alpaca")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"



# source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun6.sh
# model='70b_7b_68m'
# fn_name='sp_dy_gamma_etp_hrchl'
# start_i=5
# end_i=5
# use_dy_gamma=true
# cal_entropy=true
# start_num_data=0
# datasets_name=("mt_bench")
# # Call the run_experiments function with the input parameters
# run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"



source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun6.sh
model='70b_7b_68m'
fn_name='sp_dy_gamma_etp_hrchl'
start_i=10
end_i=10
use_dy_gamma=false
cal_entropy=false
start_num_data=0
datasets_name=("gsm8k")
# Call the run_experiments function with the input parameters
run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data "${datasets_name[@]}"





