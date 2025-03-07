
source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_00000.sh
model='13b'
fn_name='at'
start_i=10
end_i=10
use_dy_gamma=true
cal_entropy=true
temperature=1
start_num_data=0
datasets_name=("mt_bench")
# Call the run_experiments function with the input parameters
run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"

