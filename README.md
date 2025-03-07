## Usage
### Inference
You need prepare a pair of models using the same embedding and vocabulary. The approximation model should be smaller than the target model. Here are some
tested model pairs.


</center>

In the sample, we demostrate llama-70b as the target model, llama-7b as the approximation model. 

```bash
source /work/valex1377/LLMSpeculativeSampling/scripts/common_inference_fun10.sh
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


### Serving
Start an inference server.
```bash
python serving.py
```

Test the serving with curl:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Who is the president of the USA"}' http://127.0.0.1:5000/predict
```

```

## Limitations
Currently, I only support request of batch size as 1.
Since this repo is built for demostration purpose, other optimizations, such as batching and parallelism, are not included which are essential for efficiency.
=======

