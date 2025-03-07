# common_functions.sh

# Function to set target and approx model based on the input model
set_model_names() {
    local model=$1
    local dataset_name=$2
    local use_dy_gamma=$3
    local cal_entropy=$4

    
    model_70b_68m=""
    if [[ "$model" == "70b_7b_68m" || "$model" == "70b_7b_68m_adv" || "$model" == "70b_68m" || "$model" == "70b" ]]; then
        target_model='llama-2-70b-chat'
        approx_model='llama-68m'
        model_70b_68m="--model_70b_68m"
    elif [[ "$model" == "70b_7b_1b" || "$model" == "70b_7b_1b_adv" || "$model" == "70b_1b" ]]; then
        target_model='llama-2-70b-chat'
        approx_model='llama-tiny-chat'
        # model_70b_68m="--model_70b_68m"
    elif [[ "$model" == "68m_7b_68m" || "$model" == "68m_7b_68m_adv" ]]; then
        target_model='llama-68m'
        approx_model='llama-68m'
        # model_70b_68m="--model_70b_68m"        
    elif [[ "$model" == "70b_7b" ]]; then
        target_model='llama-2-70b-chat'
        approx_model='llama-2-7b-chat'        
    elif [[ "$model" == "7b_68m" || "$model" == "7b" ]]; then
        target_model='llama-2-7b-chat'
        approx_model='llama-68m'
    elif [[ "$model" == "7b_1b" ]]; then
        target_model='llama-2-7b-chat'
        approx_model='llama-tiny-chat'      
    elif [[ "$model" == "1b_68m" || "$model" == "1b" ]]; then
        target_model='llama-tiny-chat'
        approx_model='llama-68m'     
    elif [[ "$model" == "68m" ]]; then
        target_model='llama-68m'
        approx_model='llama-68m'             
    elif [[ "$model" == "68m_68m" || $model == "68m_7b_68m_adv" ]]; then
        target_model='llama-68m'
        approx_model='llama-68m'                    
    else
        echo "Model name is wrong"
        exit 1
    fi

    dataset_num_samples_arg=""
    if [[ "$dataset_name" == "gsm8k" || "$dataset_name" == "alpaca" ]]; then
        dataset_num_samples_arg="--dataset_num_samples 150"
    fi

    use_dy_gamma_arg=""
    use_dy_gamma_suffix=""
    if [ "$use_dy_gamma" = true ]; then
        use_dy_gamma_arg="--use_dy_gamma"
        use_dy_gamma_suffix="_use_dy_gamma"
    fi

    cal_entropy_arg=""
    use_dy_gamma_suffix=""
    if [ "$cal_entropy" = true ]; then
        cal_entropy_arg="--cal_entropy"
    fi


}

# Function to run experiments based on input parameters
run_experiments() {
    local model=$1
    local fn_name=$2
    local start_i=$3
    local end_i=$4
    local use_dy_gamma=$5
    local cal_entropy=$6
    local start_num_data=$7
    local datasets_name=("${@:8}")
    local root_path='/work/valex1377/LLMSpeculativeSampling/new_experiments'

    for dataset_name in "${datasets_name[@]}"
    do 
        # Call the function to set model names
        set_model_names $model $dataset_name $use_dy_gamma $cal_entropy

        for i in $(seq $start_i $end_i)
        do
            echo "=========$dataset_name + $fn_name Start======${i}_${model}_topkp0_fp16_2048tok${use_dy_gamma_suffix}==========="
            
            python main_modify.py \
                --input "The quick brown fox jumps over the lazy " \
                --file_root $root_path/${dataset_name}/${i}_${model}_topkp0_fp16_2048tok${use_dy_gamma_suffix} \
                --target_model_name $target_model \
                --approx_model_name $approx_model \
                --dataset_name ${dataset_name} \
                --test_times 3 \
                -r --record_time \
                --entropy_th 0 \
                --gamma $i \
                --top_p 0.9 --top_k 20 \
                -s 123 \
                --fn_name $fn_name \
                --max_tokens 2048 \
                --load_bits 16 \
                --prefix_file_name ${i}_${model}_topkp0_fp16_2048tok \
                $dataset_num_samples_arg \
                $model_70b_68m \
                $use_dy_gamma_arg \
                $cal_entropy_arg \
                --start_num_data $start_num_data \
                -m 0 
            
            echo "=========$dataset_name END  =====${i}_${model}_topkp20_fp16_2048tok${use_dy_gamma_suffix}==========="                    
        done
    done
}
