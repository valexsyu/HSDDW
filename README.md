## Usage
### Inference
You need prepare a pair of models using the same embedding and vocabulary. The approximation model should be smaller than the target model. Here are some
tested model pairs.


</center>

In the sample, we demostrate SP . llama-70b as the target model, llama-7b as the approximation model. 

```bash
source Hscripts/common_inference_fun10.sh
model='70b_7b'
fn_name='sp'
start_i=10
end_i=10
use_dy_gamma=false
cal_entropy=false
temperature=1
start_num_data=0
datasets_name=("alpaca")
# Call the run_experiments function with the input parameters
run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"
```


In the sample, we demostrate HSDDW . llama-70b as the target model, llama-7b as the approximation model. 

```bash
source scripts/common_inference_fun7.sh
model='70b_7b_68m_adv'
fn_name='sp_dy_gamma_etp_hrchl_adv'
start_i=10
end_i=10
use_dy_gamma=true
cal_entropy=true
temperature=1
start_num_data=0
datasets_name=("gsm8k")
# Call the run_experiments function with the input parameters
run_experiments $model $fn_name $start_i $end_i $use_dy_gamma $cal_entropy $start_num_data $temperature "${datasets_name[@]}"
```

## Limitations
Currently, I only support request of batch size as 1.
Since this repo is built for demostration purpose, other optimizations, such as batching and parallelism, are not included which are essential for efficiency.
=======

