# common_functions.sh

# Function to set target and approx model based on the input model
set_model_names() {
    local model=$1
    local dataset_name=$2
    model_70b_68m=""
    if [[ "$model" == "70b_7b_68m" || "$model" == "70b_7b_68m_adv" || "$model" == "70b_68m" ]]; then
        target_model='llama-2-70b-chat'
        approx_model='llama-68m'
        model_70b_68m="--model_70b_68m"
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

    dataset_num_samples_arg=""
    if [ "$dataset_name" != "humaneval" ]; then
        dataset_num_samples_arg="--dataset_num_samples 150"
    fi
}

# Function to run experiments based on input parameters
run_experiments() {
    local model=$1
    local fn_name=$2
    local start_i=$3
    local end_i=$4
    local datasets_name=("${@:5}")
    local root_path='/work/valex1377/LLMSpeculativeSampling/experiments'

    for dataset_name in "${datasets_name[@]}"
    do 
        # Call the function to set model names
        set_model_names $model $dataset_name

        for i in $(seq $start_i $end_i)
        do
            echo "=============================$dataset_name + $fn_name Start======================================"
            echo "==============sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok==========="
            
            python main_modify.py \
                --input "The quick brown fox jumps over the lazy " \
                --file_root $root_path/${dataset_name}/sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok \
                --target_model_name $target_model \
                --approx_model_name $approx_model \
                --dataset_name ${dataset_name} \
                --test_times 3 \
                -r --record_time \
                --entropy_th 0 \
                --gamma $i \
                --top_p 0 --top_k 0 \
                -s 123 \
                --fn_name $fn_name \
                --max_tokens 2048 \
                --load_bits 16 \
                --prefix_file_name sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok \
                $dataset_num_samples_arg \
                $model_70b_68m \
                -m 0 
            
            echo "=============================$dataset_name END======================================"
            echo "==============sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok==========="                    
        done
    done
}
