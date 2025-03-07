# common_functions.sh

# Function to set target and approx model based on the input model
set_model_names() {
    local model=$1
    local dataset_name=$2
    local record_only=$3
    model_70b_68m=""
    if [[ "$model" == "70b_7b_68m" || "$model" == "70b_7b_68m_adv" || "$model" == "70b_68m" ]]; then
        target_model='llama-2-70b-chat'
        approx_model='llama-68m'
        model_70b_68m="--model_70b_68m"
    elif [[ "$model" == "70b_7b_1b" || "$model" == "70b_7b_1b_adv" || "$model" == "70b_1b" ]]; then
        target_model='llama-2-70b-chat'
        approx_model='llama-tiny-chat'
        model_70b_68m="--model_70b_68m"
    elif [[ "$model" == "70b_7b" ]]; then
        target_model='llama-2-70b-chat'
        approx_model='llama-2-7b-chat'        
    elif [[ "$model" == "7b_68m" ]]; then
        target_model='llama-2-7b-chat'
        approx_model='llama-68m'
    elif [[ "$model" == "7b_1b" ]]; then
        target_model='llama-2-7b-chat'
        approx_model='llama-tiny-chat'        
    else
        echo "Model name is wrong"
        exit 1
    fi

    dataset_num_samples_arg=""
    if [ "$dataset_name" != "humaneval" ]; then
        dataset_num_samples_arg="--dataset_num_samples 150"
    fi

    record_only_arg=""
    record_only_suffix=""
    if [ "$record_only" = true ]; then
        r_only_arg="--record_only"
        record_only_suffix="_record_only"
    fi


}

# Function to run experiments based on input parameters
run_experiments() {
    local model=$1
    local fn_name=$2
    local start_i=$3
    local end_i=$4
    local record_only=$5
    local datasets_name=("${@:6}")
    local root_path='/work/valex1377/LLMSpeculativeSampling/experiments'

    for dataset_name in "${datasets_name[@]}"
    do 
        # Call the function to set model names
        set_model_names $model $dataset_name $record_only

        for i in $(seq $start_i $end_i)
        do
            echo "=============================$dataset_name + $fn_name Start======================================"
            echo "==============sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok${record_only_suffix}==========="
            
            python main_modify.py \
                --input "The quick brown fox jumps over the lazy " \
                --file_root $root_path/${dataset_name}/sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok${record_only_suffix} \
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
                $record_only_arg \
                -m 0 
            
            echo "=============================$dataset_name END======================================"
            echo "==============sp_dy_gamma_${i}_${model}_topkp0_fp16_2048tok${record_only_suffix}==========="                    
        done
    done
}
